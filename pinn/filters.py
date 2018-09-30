"""
    Filters are layers without trainable variables
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import tensorflow as tf
import numpy as np


class sparse_node(dict):
    """Sparse node class for use with PiNN
    Node inherits dictionary so that it is can be used as a 'fetches' and runned
    - Init with a dense tesnor and a mask
    - Init with indices, sparse value and shape
    - Sparse value can be updated
    - Dense value is constructed while needed
    """

    def __init__(self, dense=None, mask=None,
                 indices=None, sparse=None):

        if dense is not None:
            indices = tf.where(mask)
            sparse = tf.gather_nd(dense, indices)

        self.indices = indices
        self.sparse = sparse
        self.dense = dense
        self.mask = mask

    def get_dense(self):
        # Construct the dense tensor while needed
        if self.dense is None:
            self.dense = tf.scatter_nd(self.indices, self.sparse,
                                       list(self.mask.shape)+list(self.sparse.shape[1:]))
        return self.dense

    def new_nodes(self, new_sparse):
        return sparse_node(mask=self.mask,
                           indices=self.indices,
                           sparse=new_sparse)


class atomic_mask():
    """Atomic filter
    Boolean for existing atoms
    """

    def parse(self, tensors, dtype):
        atoms = tensors['atoms']
        coord = tensors['coord']
        mask = tf.cast(atoms, tf.bool)
        tensors['atoms'] = sparse_node(dense=atoms, mask=mask)


class atomic_dress():
    """Atomic dress
    Assign an energy for each type of atom
    """

    def __init__(self, dress):
        self.dress = dress

    def parse(self, tensors, dtype):
        atoms = tensors['atoms']

        sparse = 0.0

        for key, val in self.dress.items():
            sparse += tf.cast(tf.equal(atoms.sparse, key), dtype)*val

        sparse = tf.SparseTensor(atoms.indices, sparse, atoms.mask.shape)
        energy = tf.sparse_reduce_sum(sparse, [-1])

        if 'e_data' in tensors:
            # We are in training
            tensors['e_data'] -= energy
            tensors['e_data'] *= 627.509
        tensors['energy'] = tf.constant(0.0)


class pi_atomic():
    """
    Transform the atomic numbers to element property nodes
    """

    def __init__(self, types):
        self.types = types

    def parse(self, tensors, dtype):
        elem = tf.expand_dims(tensors['atoms'].sparse, -1)
        sparse = tf.concat(
            [tf.cast(tf.equal(elem, e), dtype) for e in self.types], axis=-1)
        tensors['nodes'] = {0: tensors['atoms'].new_nodes(sparse)}


class distance_mat():
    """Distance filter
    Generates a distance tensor from the coordinates
    """

    def parse(self, tensors, dtype):
        coord = tensors['coord']
        a_mask = tensors['atoms'].mask
        n = int(a_mask.shape[0])
        i = int(a_mask.shape[1])

        d_mask = (tf.expand_dims(a_mask, -1) & tf.expand_dims(a_mask, -2) & ~
                  tf.eye(i, batch_shape=[n], dtype=tf.bool))
        d_indices = tf.where(d_mask)
        coord_i = tf.gather_nd(coord, d_indices[:, 0:-1])
        coord_j = tf.gather_nd(coord, tf.concat([d_indices[:, 0:-2],
                                                 d_indices[:, -1:]], -1))
        diff = coord_i - coord_j

        if 'cell' in tensors and tensors['cell'].shape == [3]:
            diff = diff - tf.rint(diff / tensors['cell']) * tensors['cell']
        # elif tensors['cell'].shape == [3, 3]:
        # TODO Implement PBC for triclinic cells
        dist = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=-1))
        tensors['dist'] = sparse_node(
            mask=d_mask, indices=d_indices, sparse=dist)


class symm_func():
    """
    """

    def __init__(self, func='f1', rc=4.0):
        self.rc = rc
        self.func = func

    def parse(self, tensors, dtype):
        d_sparse = tensors['dist'].sparse
        d_indices = tensors['dist'].indices
        d_mask = tensors['dist'].mask

        sf_indices = tf.gather_nd(d_indices, tf.where(d_sparse < self.rc))
        sf_sparse = tf.gather_nd(d_sparse, tf.where(d_sparse < self.rc))
        sf_mask = tf.sparse_to_dense(sf_indices, d_mask.shape, True, False)

        sf_sparse = {
            'f1': lambda x: 0.5*(tf.cos(np.pi*x/self.rc)+1),
            'f2': lambda x: tf.tanh(1-x/self.rc)**3,
            'hip': lambda x: tf.cos(np.pi*x/self.rc)**2,
        }[self.func](sf_sparse)

        tensors['symm_func'] = sparse_node(mask=sf_mask,
                                           indices=sf_indices,
                                           sparse=sf_sparse)


class pi_basis():
    """

    """

    def __init__(self, func='f1', order=3, rc=4.0):
        self.rc = rc
        self.func = func
        self.order = order

    def parse(self, tensors, dtype):
        symm_func = tensors['symm_func']

        basis = tf.expand_dims(symm_func.sparse, -1)
        basis = tf.concat([basis**(i+1)
                           for i in range(self.order)], axis=-1)

        tensors['pi_basis'] = symm_func.new_nodes(basis)


class schnet_basis():
    """

    """

    def __init__(self, miu_min=0, dmiu=0.1, gamma=0.1,
                 n_basis=300, rc=30):
        self.rc = rc
        self.miumin = miu_min
        self.dmiu = dmiu
        self.gamma = gamma
        self.n_basis = n_basis

    def parse(self, tensors, dtype):
        d_sparse = tensors['dist'].sparse
        d_indices = tensors['dist'].indices
        d_mask = tensors['dist'].mask

        bf_indices = tf.gather_nd(d_indices, tf.where(d_sparse < self.rc))
        bf_sparse = tf.gather_nd(d_sparse, tf.where(d_sparse < self.rc))
        bf_mask = tf.sparse_to_dense(bf_indices, d_mask.shape, True, False)
        bf_sparse = tf.expand_dims(bf_sparse, -1)

        sparse = []
        for i in range(self.n_basis):
            miu = self.miumin + i*self.dmiu
            sparse.append(tf.exp(-self.gamma*(bf_sparse-miu)**2))
        sparse = tf.concat(sparse, axis=-1)

        tensors['pi_basis'] = sparse_node(mask=bf_mask,
                                           indices=bf_indices,
                                           sparse=sparse)


class bp_G3():
    """BP-style G3 symmetry function descriptor
    """

    def __init__(self, lambd=1, zeta=1, eta=1):
        self.lambd = lambd
        self.zeta = zeta
        self.eta = eta

    def parse(self, tensors, dtype):
        # Indices
        symm_func = tensors['symm_func']
        mask = symm_func.mask
        sf_dense = symm_func.get_dense()
        dist_dense = tensors['dist'].get_dense()

        mask_ij = tf.expand_dims(mask, -1)
        mask_ik = tf.expand_dims(mask, -2)
        mask_jk = tf.expand_dims(mask, -3)
        mask_ijk = mask_ij & mask_ik & mask_jk
        indices = tf.where(mask_ijk)

        i_ij = indices[:, 0:3]
        i_ik = tf.concat([indices[:, 0:2], indices[:, 3:]], -1)
        i_jk = tf.concat([indices[:, 0:1], indices[:, 2:4]], -1)
        # Collect
        f_ij = tf.gather_nd(sf_dense, i_ij)
        f_ik = tf.gather_nd(sf_dense, i_ik)
        f_jk = tf.gather_nd(sf_dense, i_jk)
        r_ij = tf.gather_nd(dist_dense, i_ij)
        r_ik = tf.gather_nd(dist_dense, i_ik)
        r_jk = tf.gather_nd(dist_dense, i_jk)
        # Calculate
        lambd = self.lambd
        zeta = self.zeta
        eta = self.eta
        cosin = (r_ij**2+r_jk**2-r_jk**2)/(r_ij*r_jk*2)
        gauss = tf.exp(-eta*(r_ij**2+r_jk**2+r_jk**2))
        G3 = (1+lambd*cosin)**zeta*gauss*f_ij*f_jk*f_ik
        # Reshape
        #G3 = tf.SparseTensor(indices, G3, mask_ijk.shape)
        G3 = tf.sparse_to_dense(indices, mask_ijk.shape, G3)
        G3 = 2**(1-zeta)*tf.reduce_sum(G3, axis=[-1, -2])
        G3 = tf.expand_dims(G3, -1)
        if 'bp_sf' in tensors:
            tensors['bp_sf'] = tf.concat([tensors['bp_sf'], G3], -1)
        else:
            tensors['bp_sf'] = G3


class bp_G2():
    """BP-style G2 symmetry function descriptor
    """

    def __init__(self, rs=2.0, etta=1):
        self.rs = rs
        self.etta = etta

    def parse(self, tensors, dtype):
        symm_func = tensors['symm_func']
        dist = tf.gather_nd(tensors['dist'].get_dense(), symm_func.indices)
        sf = symm_func.sparse
        G2 = tf.exp(-self.etta*(dist-self.rs)**2)*sf
        G2 = tf.reduce_sum(symm_func.new_nodes(G2).get_dense(),
                           axis=-1, keepdims=True)
        if 'bp_sf' in tensors:
            tensors['bp_sf'] = tf.concat([tensors['bp_sf'], G2], -1)
        else:
            tensors['bp_sf'] = G2

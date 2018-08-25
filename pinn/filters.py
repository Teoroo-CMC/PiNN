"""
    Filters are layers without trainable variables
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import tensorflow as tf
import numpy as np

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
        mask = tensors['sf_mask']
        mask_ij = tf.expand_dims(mask, -1)
        mask_ik = tf.expand_dims(mask, -2)
        mask_jk = tf.expand_dims(mask, -3)
        mask_ijk = mask_ij & mask_ik & mask_jk
        indices = tf.where(mask_ijk)
        i_ij =  indices[:,0:3]
        i_ik =  tf.concat([indices[:,0:2], indices[:,3:]],-1)
        i_jk =  tf.concat([indices[:,0:1], indices[:,2:4]],-1)
        # Collect
        f_ij = tf.gather_nd(tensors['bp_sf'], i_ij)
        f_ik = tf.gather_nd(tensors['bp_sf'], i_ik)
        f_jk = tf.gather_nd(tensors['bp_sf'], i_jk)
        r_ij = tf.gather_nd(tensors['dist'], i_ij)
        r_ik = tf.gather_nd(tensors['dist'], i_ik)
        r_jk = tf.gather_nd(tensors['dist'], i_jk)
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
    """BP-style G4 symmetry function descriptor
    """
    def __init__(self, rs=2.0, etta=1):
        self.rs = rs
        self.etta = etta

    def parse(self, tensors, dtype):
        symm_func = tensors['symm_func']
        dist = tensors['dist']
        G2 = tf.reduce_sum((tf.exp(-self.etta*(dist-self.rs)**2))*symm_func,
                              axis=-1, keepdims=True)

        if 'bp_sf' in tensors:
            tensors['bp_sf'] = tf.concat([tensors['bp_sf'], G2], -1)
        else:
            tensors['bp_sf'] = G2


class symm_func():
    """
    """
    def __init__(self, func='f1', rc=4.0):
        self.rc = rc
        self.func = func

    def parse(self, tensors, dtype):
        dist = tensors['dist']
        tensors['sf_mask'] = (dist<self.rc) & (dist>0)

        symm_func = {
            'f1': lambda x: 0.5*(tf.cos(np.pi*x/self.rc)+1),
            'f2': lambda x: tf.tanh(1-x/self.rc)**3
        }[self.func](dist)
        symm_func = tf.where(tensors['sf_mask'],
                             symm_func, tf.zeros_like(symm_func))

        tensors['symm_func'] = symm_func


class atomic_mask():
    """Atomic filter

    Boolean for existing atoms
    """

    def parse(self, tensors, dtype):
        a_mask = tf.cast(tensors['atoms'], tf.bool)
        tensors['a_mask'] = a_mask


class atomic_dress():
    """Atomic dress

    Assign an energy for each type of atom
    """

    def __init__(self, dress):
        self.dress = dress

    def parse(self, tensors, dtype):
        atoms = tensors['atoms']
        energy = tf.reduce_sum([
            tf.where(tf.equal(atoms, key),
                     val * tf.cast(tf.ones_like(atoms), dtype),
                     tf.cast(tf.zeros_like(atoms), dtype))
            for key, val in self.dress.items()], [-1, 0])
        if 'e_data' in tensors:
            tensors['e_data'] -= energy
            tensors['e_data'] *= 627.509
        tensors['energy'] = tf.constant(0.0)


class distance_mat():
    """Distance filter
    Generates a distance tensor from the coordinates
    """

    def parse(self, tensors, dtype):
        coord = tensors['coord']
        diff = tf.expand_dims(coord, -2) - tf.expand_dims(coord, -3)

        if 'cell' in tensors and tensors['cell'].shape == [3]:
            diff = diff - tf.rint(diff / tensors['cell']) * tensors['cell']
        # elif tensors['cell'].shape == [3, 3]:
        #TODO Implement PBC for triclinic cells

        square = tf.reduce_sum(tf.square(diff), axis=-1)
        # To make the distance differentiable
        zeros = tf.equal(square, 0)
        square = tf.where(zeros,  tf.ones_like(square), square)
        dist = tf.where(zeros, tf.zeros_like(square), tf.sqrt(square))
        tensors['dist'] = dist


class pi_atomic():
    """

    """

    def __init__(self, types):
        self.types = types

    def parse(self, tensors, dtype):
        shape = [1] * (len(tensors['atoms'].shape))
        shape.append(len(self.types))
        types = tf.reshape(self.types, shape)
        atoms = tf.expand_dims(tensors['atoms'], -1)
        p_nodes = tf.cast(tf.equal(types, atoms),
                          dtype)

        tensors['nodes'] = {}
        tensors['nodes'][0] = p_nodes


class pi_kernel():
    """

    """

    def __init__(self, func='f1', order=2, rc=4.0):
        self.rc = rc
        self.func = func
        self.order = order

    def parse(self, tensors, dtype):
        dist = tensors['dist']
        symm_func = tf.expand_dims(dist, -1)
        symm_func = {
            'f1': lambda x: 0.5*(tf.cos(np.pi*symm_func/self.rc)+1),
            'f2': lambda x: tf.tanh(1-symm_func/self.rc) ** 3
        }[self.func](symm_func)

        kernel = tf.concat([symm_func**(i+1)
                            for i in range(self.order)], axis=-1)

        kernel = tf.expand_dims(kernel, -2)
        p_mask = tf.expand_dims(tf.cast(tensors['a_mask'], dtype), -1)
        i_mask = tf.expand_dims(
            tf.cast((dist > 0) & (dist < self.rc), dtype), -1)
        i_mask = i_mask * tf.expand_dims(p_mask, -2) *tf.expand_dims(p_mask, -3)

        tensors['pi_kernel'] = {1: kernel}
        tensors['pi_masks'] = {0: p_mask, 1: i_mask}

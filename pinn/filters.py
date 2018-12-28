# -*- coding: utf-8 -*-
"""Filters are special layers that does not contain trainable vairables

To ease the preprocessing, all filters act on a tensor (nested) dictionary.
Note that the filters return a function, which accepts a nested tensor object
and adds certain keys.
"""

import numpy as np
import tensorflow as tf
from functools import wraps

def pinn_filter(func):
    @wraps(func)
    def filter_wrapper(*args, **kwargs):
        return lambda t: func(t, *args, **kwargs)
    return filter_wrapper

@pinn_filter
def sparsify(tensors):
    """ Sparsity atomic inputs

    ind and nl are sparse representations of connections.
    From n -> n + 1 order
       0: image -> 1: atom -> 2: pair -> 3: triplet (pair of pairs)...
    ind[n] is a i*j tensor where
       i = number of order n elements,
       j = number of entries defining a element
       each entry is the index of n-1 order element in ind[n-1]
    nl[n] is a i*j tensor where:
       i = the number of n-1 level elements
       j = max number of neighbors
       each non-zero entry is the index of i's neighbour plus one
    """
    atom_ind = tf.cast(tf.where(tensors['atoms']), tf.int32)
    ind_1 = atom_ind[:,:1]
    ind_sp = tf.cumsum(tf.ones(tf.shape(ind_1), tf.int32))-1
    tensors['ind'] = {1: ind_1}
    elem = tf.gather_nd(tensors['atoms'], atom_ind)
    coord = tf.gather_nd(tensors['coord'], atom_ind)
    tensors['elem'] = elem
    tensors['coord'] = coord
    if 'f_data' in tensors:
        tensors['f_data'] = tf.gather_nd(tensors['f_data'], atom_ind)

def _displace_matrix(max_repeat):
    d = []
    n_repeat = max_repeat*2 + 1
    tot_repeat = tf.reduce_prod(n_repeat)
    for i in range(3):
        d.append(tf.cumsum(tf.ones(n_repeat, tf.int32), axis=i)
                 - max_repeat[i] -1)
    d = tf.reshape(tf.stack(d, axis=-1), [tot_repeat, 3])
    d = tf.concat([d[:tot_repeat//2] ,d[tot_repeat//2+1:]],0)
    return d


def _pbc_repeat(tensors, rc):
    n_repeat = rc * tf.norm(tf.matrix_inverse(tensors['cell']),axis=1)
    n_repeat = tf.cast(tf.ceil(n_repeat), tf.int32)
    max_repeat = tf.reduce_max(n_repeat, axis=0)
    disp_mat = _displace_matrix(max_repeat)

    repeat_mask = tf.reduce_all(
        tf.expand_dims(n_repeat,1)>=tf.abs(disp_mat),axis=2)
    atom_mask = tf.gather(repeat_mask, tensors['ind'][1])
    repeat_ar = tf.cast(tf.where(atom_mask), tf.int32)
    repeat_a = repeat_ar[:,:1]
    repeat_r = repeat_ar[:,2]
    repeat_s = tf.gather_nd(tensors['ind'][1], repeat_a)
    repeat_pos = (tf.gather_nd(tensors['coord'], repeat_a) +
                  tf.reduce_sum(
                      tf.gather_nd(tensors['cell'], repeat_s) *
                      tf.gather(tf.cast(tf.expand_dims(disp_mat,2),
                                        tf.float32), repeat_r),1))
    return repeat_pos, repeat_s, repeat_a


@pinn_filter
def cell_list_nl(tensors, rc=5.0):
    """ Compute neighbour list with celllist approach
    https://en.wikipedia.org/wiki/Cell_lists
    This is very lengthy and confusing implementation of cell list nl.
    Probably needs optimization outside Tensorflow.
    """
    atom_sind = tensors['ind'][1]
    atom_apos = tensors['coord']
    atom_gind = tf.cumsum(tf.ones_like(atom_sind), 0)
    atom_aind = atom_gind - 1
    to_collect = atom_aind
    if 'cell' in tensors:
        rep_apos, rep_sind, rep_aind = _pbc_repeat(tensors, rc)
        atom_sind = tf.concat([atom_sind, rep_sind], 0)
        atom_apos = tf.concat([atom_apos, rep_apos], 0)
        atom_aind = tf.concat([atom_aind, rep_aind], 0)
        atom_gind = tf.cumsum(tf.ones_like(atom_sind), 0)
        
    atom_apos = atom_apos - tf.reduce_min(atom_apos, axis=0)
    atom_cpos = tf.concat([atom_sind, tf.cast(atom_apos//rc, tf.int32)],axis=1)
    cpos_shap = tf.concat([tf.reduce_max(atom_cpos, axis=0) + 1,[1]], axis=0)
    samp_ccnt = tf.squeeze(tf.scatter_nd(
        atom_cpos, tf.ones_like(atom_sind, tf.int32), cpos_shap), axis=-1)
    cell_cpos = tf.cast(tf.where(samp_ccnt), tf.int32)
    cell_cind = tf.cumsum(tf.ones(tf.shape(cell_cpos)[0], tf.int32))
    cell_cind = tf.expand_dims(cell_cind, 1)
    samp_cind = tf.squeeze(tf.scatter_nd(
        cell_cpos, cell_cind, cpos_shap), axis=-1)
    # Get the atom's relative index(rind) and position(rpos) in cell
    # And each cell's atom list (alst)
    atom_cind = tf.gather_nd(samp_cind, atom_cpos) - 1
    atom_cind_args = tf.contrib.framework.argsort(atom_cind, axis=0)
    atom_cind_sort = tf.gather(atom_cind, atom_cind_args)
        
    atom_rind_sort = tf.cumsum(tf.ones_like(atom_cind, tf.int32))
    cell_rind_min  = tf.segment_min(atom_rind_sort, atom_cind_sort)
    atom_rind_sort = atom_rind_sort - tf.gather(cell_rind_min, atom_cind_sort)
    atom_rpos_sort = tf.stack([atom_cind_sort, atom_rind_sort],axis=1)
    atom_rpos = tf.unsorted_segment_sum(atom_rpos_sort, atom_cind_args,
                                        tf.shape(atom_gind)[0])
    cell_alst_shap = [tf.shape(cell_cind)[0], tf.reduce_max(samp_ccnt),1]
    cell_alst = tf.squeeze(tf.scatter_nd(
        atom_rpos, atom_gind, cell_alst_shap), axis=-1)
    # Get cell's linked cell list, for cells in to_collect only
    disp_mat = np.zeros([3,3,3,4], np.int32)
    disp_mat[:,:,:,1] = np.reshape([-1,0,1], (3,1,1))
    disp_mat[:,:,:,2] = np.reshape([-1,0,1], (1,3,1))
    disp_mat[:,:,:,3] = np.reshape([-1,0,1], (1,1,3))
    disp_mat = np.reshape(disp_mat,(1, 27, 4))
    cell_npos = tf.expand_dims(cell_cpos,1) + disp_mat
    npos_mask = tf.reduce_all((cell_npos>=0) & (cell_npos < cpos_shap[:-1]), 2)
    cell_nind = tf.squeeze(tf.scatter_nd(
        tf.cast(tf.where(npos_mask), tf.int32), 
        tf.expand_dims(tf.gather_nd(
            samp_cind, tf.boolean_mask(cell_npos, npos_mask)),1),
        tf.concat([tf.shape(cell_npos)[:-1],[1]],0)),-1)
    # Finally, a sparse list of atom pairs
    coll_nind = tf.gather(cell_nind, tf.gather_nd(atom_cind, to_collect))
    pair_ic = tf.cast(tf.where(coll_nind), tf.int32)
    pair_ic_i = pair_ic[:,0]
    pair_ic_c = tf.gather_nd(coll_nind, pair_ic) - 1
    pair_ic_alst = tf.gather(cell_alst, pair_ic_c)

    pair_ij = tf.cast(tf.where(pair_ic_alst), tf.int32)
    pair_ij_i = tf.gather(pair_ic_i, pair_ij[:,0])
    pair_ij_j = tf.gather_nd(pair_ic_alst, pair_ij) - 1
    
    diff = tf.gather(atom_apos, pair_ij_j) - tf.gather(atom_apos, pair_ij_i)
    dist = tf.norm(diff, axis=-1)
    ind_rc = tf.where((dist<rc) & (dist>0))
    dist = tf.gather_nd(dist, ind_rc)
    diff = tf.gather_nd(diff, ind_rc)
    pair_i_aind = tf.gather_nd(tf.gather(atom_aind, pair_ij_i), ind_rc)
    pair_j_aind = tf.gather_nd(tf.gather(atom_aind, pair_ij_j), ind_rc)
    tensors['ind'][2] = tf.concat([pair_i_aind, pair_j_aind], 1)
    tensors['dist'] = dist
    tensors['diff'] = diff


@pinn_filter
def naive_nl(tensors, rc=5.0):
    """ Construct pairs by calculating all possible pairs, without PBC
    """
    # Naive nl_1, will be abandoned
    ind_1 = tensors['ind'][1]
    coord = tensors['coord']
    atom_ind = tf.cast(tf.where(tensors['atoms']), tf.int32)
    nl_1 = tf.scatter_nd(atom_ind, tf.squeeze(ind_sp)+1,
                         tf.shape(tensors['atoms'],
                                  out_type=tf.int32))
    nl_2 = tf.gather_nd(nl_1, ind_1)
    pair_ind = tf.cast(tf.where(nl_2), tf.int32)
    ind_2_i = pair_ind[:, 0]
    ind_2_j = tf.gather_nd(nl_2, pair_ind)-1
    ind_2 = tf.stack([ind_2_i, ind_2_j], axis=1)
    # Gathering the interacting indices
    diff = tf.gather(coord, ind_2_j) - tf.gather(coord, ind_2_i)
    dist = tf.sqrt((tf.reduce_sum(tf.square(diff), axis=-1)))
    ind_rc = tf.where((dist>0) & (dist<rc))
    # The gradient of this sparse diff is masked,
    diff = tf.gather_nd(diff, ind_rc)
    dist = tf.gather_nd(dist, ind_rc)
    # Rewire the back-prop, the displacement can be differentiated now
    # Todo: this should be handeled by models after we implement
    #       the rewiring of following derivitives during preprocessing:
    #       coord -> diff -> dist -> symm_func -> basis
    #       so that we can preprocess while training forces
    tensors['diff'] = diff
    tensors['dist'] = dist
    tensors['ind'][2] = tf.gather_nd(ind_2, ind_rc)


@pinn_filter
def atomic_dress(tensors, dress, dtype=tf.float32):
    """Assign an energy to each specified elems

    Args:
        dress (dict): dictionary consisting the atomic energies
    """
    elem = tensors['elem']
    e_dress = tf.zeros_like(elem, dtype)
    for k, val in dress.items():
        indices = tf.cast(tf.equal(elem, k), dtype)
        e_dress += indices * tf.cast(val, dtype)
    n_batch = tf.shape(tensors['atoms'])[0]
    e_dress = tf.unsorted_segment_sum(
        e_dress, tensors['ind'][1][:,0], n_batch)
    tensors['e_dress'] = tf.squeeze(e_dress)

@pinn_filter
def symm_func(tensors, sf_type='f1', rc=5.0):
    """Adds the symmetry function of given type

    Args:
        sf_type (string): name of the symmetry function
        rc: cutoff radius
    """
    dist = tensors['dist']
    sf = {'f1': lambda x: 0.5*(tf.cos(np.pi*x/rc)+1),
          'f2': lambda x: tf.tanh(1-x/rc)**3,
          'hip': lambda x: tf.cos(np.pi*x/rc)**2}
    tensors['symm_func'] = sf[sf_type](dist)

@pinn_filter
def pi_basis(tensors, order=4):
    """ Adds PiNN stype basis function for interation

    Args:
        order (int): the order of the polynomial expansion
    """
    symm_func = tensors['symm_func']
    basis = tf.expand_dims(symm_func, -1)
    basis = tf.concat(
        [basis**(i+1) for i in range(order)], axis=-1)
    tensors['pi_basis'] = tf.expand_dims(basis,-2)

@pinn_filter
def atomic_onehot(tensors, atom_types=[1,6,7,8,9],
                  dtype=tf.float32):
    """ Perform one-hot encoding on elements

    Args:
        atom_types (list): elements to encode
        dtype: dtype for the output
    """
    output = tf.equal(tf.expand_dims(tensors['elem'],1),
                      tf.expand_dims(atom_types, 0))
    output = tf.cast(output, dtype)
    tensors['elem_onehot'] = output




# class schnet_basis():
#     """

#     """

#     def __init__(self, miu_min=0, dmiu=0.1, gamma=0.1,
#                  n_basis=300, rc=30):
#         self.rc = rc
#         self.miumin = miu_min
#         self.dmiu = dmiu
#         self.gamma = gamma
#         self.n_basis = n_basis

#     def parse(self, tensors, dtype):
#         d_sparse = tensors['dist'].sparse
#         d_indices = tensors['dist'].indices
#         d_mask = tensors['dist'].mask

#         bf_indices = tf.gather_nd(d_indices, tf.where(d_sparse < self.rc))
#         bf_sparse = tf.gather_nd(d_sparse, tf.where(d_sparse < self.rc))
#         bf_mask = tf.sparse_to_dense(bf_indices, d_mask.shape, True, False)
#         bf_sparse = tf.expand_dims(bf_sparse, -1)

#         sparse = []
#         for i in range(self.n_basis):
#             miu = self.miumin + i*self.dmiu
#             sparse.append(tf.exp(-self.gamma*(bf_sparse-miu)**2))
#         sparse = tf.concat(sparse, axis=-1)

#         tensors['pi_basis'] = sparse_node(mask=bf_mask,
#                                            indices=bf_indices,
#                                            sparse=sparse)


# class bp_G3():
#     """BP-style G3 symmetry function descriptor
#     """

#     def __init__(self, lambd=1, zeta=1, eta=1):
#         self.lambd = lambd
#         self.zeta = zeta
#         self.eta = eta

#     def parse(self, tensors, dtype):
#         # Indices
#         symm_func = tensors['symm_func']
#         mask = symm_func.mask
#         sf_dense = symm_func.get_dense()
#         dist_dense = tensors['dist'].get_dense()

#         mask_ij = tf.expand_dims(mask, -1)
#         mask_ik = tf.expand_dims(mask, -2)
#         mask_jk = tf.expand_dims(mask, -3)
#         mask_ijk = mask_ij & mask_ik & mask_jk
#         indices = tf.where(mask_ijk)

#         i_ij = indices[:, 0:3]
#         i_ik = tf.concat([indices[:, 0:2], indices[:, 3:]], -1)
#         i_jk = tf.concat([indices[:, 0:1], indices[:, 2:4]], -1)
#         # Collect
#         f_ij = tf.gather_nd(sf_dense, i_ij)
#         f_ik = tf.gather_nd(sf_dense, i_ik)
#         f_jk = tf.gather_nd(sf_dense, i_jk)
#         r_ij = tf.gather_nd(dist_dense, i_ij)
#         r_ik = tf.gather_nd(dist_dense, i_ik)
#         r_jk = tf.gather_nd(dist_dense, i_jk)
#         # Calculate
#         lambd = self.lambd
#         zeta = self.zeta
#         eta = self.eta
#         cosin = (r_ij**2+r_jk**2-r_jk**2)/(r_ij*r_jk*2)
#         gauss = tf.exp(-eta*(r_ij**2+r_jk**2+r_jk**2))
#         G3 = (1+lambd*cosin)**zeta*gauss*f_ij*f_jk*f_ik
#         # Reshape
#         #G3 = tf.SparseTensor(indices, G3, mask_ijk.shape)
#         G3 = tf.sparse_to_dense(indices, mask_ijk.shape, G3)
#         G3 = 2**(1-zeta)*tf.reduce_sum(G3, axis=[-1, -2])
#         G3 = tf.expand_dims(G3, -1)
#         if 'bp_sf' in tensors:
#             tensors['bp_sf'] = tf.concat([tensors['bp_sf'], G3], -1)
#         else:
#             tensors['bp_sf'] = G3


# class bp_G2():
#     """BP-style G2 symmetry function descriptor
#     """

#     def __init__(self, rs=2.0, etta=1):
#         self.rs = rs
#         self.etta = etta

#     def parse(self, tensors, dtype):
#         symm_func = tensors['symm_func']
#         dist = tf.gather_nd(tensors['dist'].get_dense(), symm_func.indices)
#         sf = symm_func.sparse
#         G2 = tf.exp(-self.etta*(dist-self.rs)**2)*sf
#         G2 = tf.reduce_sum(symm_func.new_nodes(G2).get_dense(),
#                            axis=-1, keepdims=True)
#         if 'bp_sf' in tensors:
#             tensors['bp_sf'] = tf.concat([tensors['bp_sf'], G2], -1)
#         else:
#             tensors['bp_sf'] = G2

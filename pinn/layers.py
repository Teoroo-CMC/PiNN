# -*- coding: utf-8 -*-
"""Layers are operations on the tensors.

Layers here should be pure functions (do not change the inputs).
"""
import numpy as np
import tensorflow as tf
from pinn.utils import pi_named


def _displace_matrix(max_repeat):
    """This is a helper function for cell_list_nl"""
    d = []
    n_repeat = max_repeat*2 + 1
    tot_repeat = tf.reduce_prod(n_repeat)
    for i in range(3):
        d.append(tf.cumsum(tf.ones(n_repeat, tf.int32), axis=i)
                 - max_repeat[i] - 1)
    d = tf.reshape(tf.stack(d, axis=-1), [tot_repeat, 3])
    d = tf.concat([d[:tot_repeat//2], d[tot_repeat//2+1:]], 0)
    return d


def _pbc_repeat(coord, cell, ind_1, rc):
    """This is a helper function for cell_list_nl"""
    n_repeat = rc * tf.norm(tf.matrix_inverse(cell), axis=1)
    n_repeat = tf.cast(tf.ceil(n_repeat), tf.int32)
    max_repeat = tf.reduce_max(n_repeat, axis=0)
    disp_mat = _displace_matrix(max_repeat)

    repeat_mask = tf.reduce_all(
        tf.expand_dims(n_repeat, 1) >= tf.abs(disp_mat), axis=2)
    atom_mask = tf.gather(repeat_mask, ind_1)
    repeat_ar = tf.cast(tf.where(atom_mask), tf.int32)
    repeat_a = repeat_ar[:, :1]
    repeat_r = repeat_ar[:, 2]
    repeat_s = tf.gather_nd(ind_1, repeat_a)
    repeat_pos = (tf.gather_nd(coord, repeat_a) +
                  tf.reduce_sum(
                      tf.gather_nd(cell, repeat_s) *
                      tf.gather(tf.cast(tf.expand_dims(disp_mat, 2),
                                        tf.float32), repeat_r), 1))
    return repeat_pos, repeat_s, repeat_a


def _wrap_coord(tensors):
    """wrap positions to unit cell"""
    cell = tf.gather_nd(tensors['cell'], tensors['ind_1'])
    coord = tf.expand_dims(tensors['coord'], -1)
    frac_coord = tf.linalg.solve(tf.transpose(cell, perm=[0, 2, 1]), coord)
    frac_coord %= 1
    coord = tf.matmul(tf.transpose(cell, perm=[0, 2, 1]), frac_coord)
    return tf.squeeze(coord, -1)


@pi_named('cell_list_nl')
def cell_list_nl(tensors, rc=5.0):
    """ Compute neighbour list with celllist approach
    https://en.wikipedia.org/wiki/Cell_lists
    This is very lengthy and confusing implementation of cell list nl.
    Probably needs optimization outside Tensorflow.

    The function expects a dictionary of tensors from a sparse_batch
    with keys: 'ind_1', 'coord' and optionally 'cell'
    """
    atom_sind = tensors['ind_1']
    atom_apos = tensors['coord']
    atom_gind = tf.cumsum(tf.ones_like(atom_sind), 0)
    atom_aind = atom_gind - 1
    to_collect = atom_aind
    if 'cell' in tensors:
        coord_wrap = _wrap_coord(tensors)
        atom_apos =  coord_wrap
        rep_apos, rep_sind, rep_aind = _pbc_repeat(
            coord_wrap, tensors['cell'], tensors['ind_1'], rc)
        atom_sind = tf.concat([atom_sind, rep_sind], 0)
        atom_apos = tf.concat([atom_apos, rep_apos], 0)
        atom_aind = tf.concat([atom_aind, rep_aind], 0)
        atom_gind = tf.cumsum(tf.ones_like(atom_sind), 0)
    atom_apos = atom_apos - tf.reduce_min(atom_apos, axis=0)
    atom_cpos = tf.concat(
        [atom_sind, tf.cast(atom_apos//rc, tf.int32)], axis=1)
    cpos_shap = tf.concat([tf.reduce_max(atom_cpos, axis=0) + 1, [1]], axis=0)
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
    cell_rind_min = tf.segment_min(atom_rind_sort, atom_cind_sort)
    atom_rind_sort = atom_rind_sort - tf.gather(cell_rind_min, atom_cind_sort)
    atom_rpos_sort = tf.stack([atom_cind_sort, atom_rind_sort], axis=1)
    atom_rpos = tf.unsorted_segment_sum(atom_rpos_sort, atom_cind_args,
                                        tf.shape(atom_gind)[0])
    cell_alst_shap = [tf.shape(cell_cind)[0], tf.reduce_max(samp_ccnt), 1]
    cell_alst = tf.squeeze(tf.scatter_nd(
        atom_rpos, atom_gind, cell_alst_shap), axis=-1)
    # Get cell's linked cell list, for cells in to_collect only
    disp_mat = np.zeros([3, 3, 3, 4], np.int32)
    disp_mat[:, :, :, 1] = np.reshape([-1, 0, 1], (3, 1, 1))
    disp_mat[:, :, :, 2] = np.reshape([-1, 0, 1], (1, 3, 1))
    disp_mat[:, :, :, 3] = np.reshape([-1, 0, 1], (1, 1, 3))
    disp_mat = np.reshape(disp_mat, (1, 27, 4))
    cell_npos = tf.expand_dims(cell_cpos, 1) + disp_mat
    npos_mask = tf.reduce_all(
        (cell_npos >= 0) & (cell_npos < cpos_shap[:-1]), 2)
    cell_nind = tf.squeeze(tf.scatter_nd(
        tf.cast(tf.where(npos_mask), tf.int32),
        tf.expand_dims(tf.gather_nd(
            samp_cind, tf.boolean_mask(cell_npos, npos_mask)), 1),
        tf.concat([tf.shape(cell_npos)[:-1], [1]], 0)), -1)
    # Finally, a sparse list of atom pairs
    coll_nind = tf.gather(cell_nind, tf.gather_nd(atom_cind, to_collect))
    pair_ic = tf.cast(tf.where(coll_nind), tf.int32)
    pair_ic_i = pair_ic[:, 0]
    pair_ic_c = tf.gather_nd(coll_nind, pair_ic) - 1
    pair_ic_alst = tf.gather(cell_alst, pair_ic_c)

    pair_ij = tf.cast(tf.where(pair_ic_alst), tf.int32)
    pair_ij_i = tf.gather(pair_ic_i, pair_ij[:, 0])
    pair_ij_j = tf.gather_nd(pair_ic_alst, pair_ij) - 1

    diff = tf.gather(atom_apos, pair_ij_j) - tf.gather(atom_apos, pair_ij_i)
    dist = tf.norm(diff, axis=-1)
    ind_rc = tf.where((dist < rc) & (dist > 0))
    dist = tf.gather_nd(dist, ind_rc)
    diff = tf.gather_nd(diff, ind_rc)
    pair_i_aind = tf.gather_nd(tf.gather(atom_aind, pair_ij_i), ind_rc)
    pair_j_aind = tf.gather_nd(tf.gather(atom_aind, pair_ij_j), ind_rc)

    output = {
        'ind_2': tf.concat([pair_i_aind, pair_j_aind], 1),
        'dist': dist,
        'diff': diff
    }
    return output


@pi_named('atomic_dress')
def atomic_dress(tensors, dress, dtype=tf.float32):
    """Assign an energy to each specified elems

    Args:
        dress (dict): dictionary consisting the atomic energies
    """
    elem = tensors['elems']
    e_dress = tf.zeros_like(elem, dtype)
    for k, val in dress.items():
        indices = tf.cast(tf.equal(elem, k), dtype)
        e_dress += indices * tf.cast(val, dtype)
    n_batch = tf.reduce_max(tensors['ind_1'])+1
    e_dress = tf.unsorted_segment_sum(
        e_dress, tensors['ind_1'][:, 0], n_batch)
    return e_dress


@pi_named('cutoff_func')
def cutoff_func(dist, cutoff_type='f1', rc=5.0):
    """returns the cutoff function of given type

    Args:
        dist (tensor): a tensor of distance
        cutoff_type (string): name of the cutoff function
        rc (float): cutoff radius

    Returns: 
        A cutoff function tensor with the same shape of dist
    """
    cutoff_fn = {'f1': lambda x: 0.5*(tf.cos(np.pi*x/rc)+1),
                 'f2': lambda x: (tf.tanh(1-x/rc)/np.tanh(1))**3,
                 'hip': lambda x: tf.cos(np.pi*x/rc/2)**2}
    return cutoff_fn[cutoff_type](dist)


@pi_named('gaussian_basis')
def gaussian_basis(dist, cutoff_type, rc, n_basis, gamma):
    """ Adds PiNN style basis function for interaction

    Args:
        dist (tensor):
        cutoff_type:
        rc: cutoff radius
        n_basis: number of gaussian functions in the interval [0, rc)
        gamma: controls width of gaussian functions

    Returns:
        a basis tensor with shape (shape_base x n_basis)
    """
    cutoff = cutoff_func(dist, cutoff_type, rc)
    centers = np.linspace(0, rc, n_basis)
    basis = tf.stack([tf.exp(-gamma*(dist-center)**2)*cutoff
                      for center in centers], axis=1)
    return basis


@pi_named('polynomial_basis')
def polynomial_basis(dist, cutoff_type, rc, n_basis=4):
    """ Adds PiNN style basis function for interaction

    Args:
        dist (tensor):
        cutoff_type:
        rc: cutoff radius
        n_basis (int): the max order of the polynomial expansion
            can be a list of int as well.

    Returns: 
        a basis tensor with shape (shape_base x n_basis)
    """
    cutoff = cutoff_func(dist, cutoff_type, rc)
    if type(n_basis) != list:
        n_basis = [(i+1) for i in range(n_basis)]
    basis = tf.stack([cutoff**(i) for i in n_basis], axis=1)
    return basis


@pi_named('atomic_onehot')
def atomic_onehot(elems, atom_types=[1, 6, 7, 8, 9], dtype=tf.float32):
    """ Perform one-hot encoding on elements

    Args:
        elems (tensor): tensor (n_atoms) of elems
        atom_types (list): elements to encode
        dtype: dtype for the output

    Returns: 
        elems (tensor): (n_atoms x n_types) embedding tensor
    """
    output = tf.equal(tf.expand_dims(elems, 1),
                      tf.expand_dims(atom_types, 0))
    output = tf.cast(output, dtype)
    return output

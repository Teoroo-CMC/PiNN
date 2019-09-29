# -*- coding: utf-8 -*-
import tensorflow as tf
from pinn.utils import pi_named, connect_dist_grad
from pinn.layers import cell_list_nl, cutoff_func


@pi_named('G2_symm_func')
def G2_SF(tensors, Rs, eta, i, j):
    """ BP-style G2 symmetry functions.

    Args:
        i: central atom type, can be "ALL".
        j: neighbor atom type, can be "ALL".
        Rs: a list of Rs values.    
        eta: a list of eta values, eta and Rs must have the same length.

    Returns:
        fp: a (n_atom x n_fingerprint) tensor of fingerprints
            where n_atom is the number of central atoms defined by "i"
        jacob: a (n_pair x n_fingerprint) tensor 
            where n_pair is the number of relavent pairs in this SF
        jacob_ind: a (n_pair) tensor 
            each row correspond to the (p_ind x i_rind) of the pair
            p_ind => the relative position of this pair within all pairs
            i_rind => the index of the central atom for this pair
    """
    R = tensors['dist']
    fc = tensors['cutoff_func']
    # Compute p_filter => boolean mask of relavent pairwise interactions
    p_filter = None
    a_filter = None
    # relative position of i atom in the "to_output" group (defined by i)
    i_rind = tensors['ind_2'][:, 0]
    a_rind = tf.cumsum(tf.ones_like(tensors['elems'], tf.int32))-1
    if i != 'ALL':
        i_elem = tf.gather(tensors['elems'], i_rind)
        p_filter = tf.equal(i_elem, i)
        a_rind = tf.cumsum(tf.cast(tf.equal(tensors['elems'], i), tf.int32))-1
    if j != 'ALL':
        j_elem = tf.gather(tensors['elems'], tensors['ind_2'][:, 1])
        j_filter = tf.equal(j_elem, j)
        p_filter = tf.reduce_all(
            [p_filter, j_filter], axis=0) if p_filter is not None else j_filter
    # Gather the interactions
    if p_filter is not None:
        p_ind = tf.cast(tf.where(p_filter)[:, 0], tf.int32)
        R = tf.gather(R, p_ind)
        fc = tf.gather(fc, p_ind)
        i_rind = tf.gather(a_rind, tf.gather(i_rind, p_ind))
    else:
        p_ind = tf.cumsum(tf.ones_like(i_rind))-1
    # Symmetry function
    n_sf = len(Rs)
    R = tf.expand_dims(R, 1)
    fc = tf.expand_dims(fc, 1)
    Rs = tf.expand_dims(Rs, 0)
    eta = tf.expand_dims(eta, 0)
    sf = tf.exp(-eta*(R-Rs)**2)*fc
    fp = tf.scatter_nd(tf.expand_dims(i_rind, 1), sf,
                       [tf.reduce_max(a_rind)+1, n_sf])
    jacob = tf.stack([tf.gradients(sf[:, i], tensors['diff'])[0]
                      for i in range(n_sf)], axis=2)
    jacob = tf.gather_nd(jacob, tf.expand_dims(p_ind, 1))
    jacob_ind = tf.stack([p_ind, i_rind], axis=1)
    return fp, jacob, jacob_ind


@pi_named('G3_symm_func')
def G3_SF(tensors, cutoff_type, rc, lambd, zeta, eta, i="ALL", j="ALL", k="ALL"):
    """BP-style G3 symmetry functions.

    NOTE(YS): here diff_jk is calculated through diff_ik - diff_ij instead of 
    retrieving the distance calcualted in cell_list_nl.
    This makes is easier to get the jacobian with grad(sf, [diff_ij, diff_ik]).
    However, this also introduce some waste of computation during the fingerprint
    generation.

    Args:
        cutoff_type, rc: cutoff function and radius
            (we need them again because diff_jk is re-calculated)
        lambd: a list of lambda values.
        zeta: a list of zeta values.
        eta: a list of eta values.
        i, j, k: atom types (as int32)

    Returns:
        fp: a (n_atom x n_fingerprint) tensor of fingerprints
            where n_atom is the number of central atoms defined by "i"
        jacob: a (n_pair x n_fingerprint) tensor 
            where n_pair is the number of relavent pairs in this SF
        jacob_ind: a (n_pair) tensor 
            each row correspond to the (p_ind, i_rind) of the pair
            p_ind => the relative position of this pair within all pairs
            i_rind => the index of the central atom for this pair    
    """
    if 'ind_3' not in tensors:
        tensors['ind_3'] = _form_triplet(tensors)

    R = tensors['dist']
    fc = tensors['cutoff_func']
    diff = tensors['diff']
    ind_ij = tensors['ind_3'][:, 0]
    ind_ik = tensors['ind_3'][:, 1]
    ind2 = tensors['ind_2']
    i_rind = tf.gather(tensors['ind_2'][:, 0], ind_ij)
    # Build triplet filter
    t_filter = None
    a_rind = tf.cumsum(tf.ones_like(tensors['elems'], tf.int32))-1
    if i != 'ALL':
        i_elem = tf.gather(tensors['elems'], tf.gather(ind2[:, 0], ind_ij))
        t_filter = tf.equal(i_elem, i)
        a_rind = tf.cumsum(tf.cast(tf.equal(tensors['elems'], i), tf.int32))-1
    if j != 'ALL':
        j_elem = tf.gather(tensors['elems'], tf.gather(ind2[:, 1], ind_ij))
        j_filter = tf.equal(j_elem, j)
        t_filter = tf.reduce_all(
            [t_filter, j_filter], axis=0) if t_filter is not None else j_filter
    if k != 'ALL':
        k_elem = tf.gather(tensors['elems'], tf.gather(ind2[:, 1], ind_ik))
        k_filter = tf.equal(k_elem, k)
        t_filter = tf.reduce_all(
            [t_filter, k_filter], axis=0) if t_filter is not None else k_filter
    if t_filter is not None:
        t_ind = tf.where(t_filter)[:, 0]
        ind_ij = tf.gather(ind_ij, t_ind)
        ind_ik = tf.gather(ind_ik, t_ind)
        i_rind = tf.gather(a_rind, tf.gather(i_rind, t_ind))
    # Filter according to R_jk, once more
    diff_ij = tf.gather_nd(diff, tf.expand_dims(ind_ij, 1))
    diff_ik = tf.gather_nd(diff, tf.expand_dims(ind_ik, 1))
    diff_jk = diff_ik - diff_ij
    R_jk = tf.norm(diff_jk, axis=1)
    t_ind = tf.where(R_jk < rc)[:, 0]
    R_jk = tf.gather(R_jk, t_ind)
    fc_jk = cutoff_func(R_jk, cutoff_type=cutoff_type, rc=rc)
    ind_ij = tf.gather(ind_ij, t_ind)
    ind_ik = tf.gather(ind_ik, t_ind)
    i_rind = tf.gather(i_rind, t_ind)
    diff_ij = tf.gather_nd(diff_ij, tf.expand_dims(t_ind, 1))
    diff_ik = tf.gather_nd(diff_ik, tf.expand_dims(t_ind, 1))
    # G3 symmetry function
    R_ij = tf.expand_dims(tf.gather(R, ind_ij), 1)
    R_ik = tf.expand_dims(tf.gather(R, ind_ik), 1)
    R_jk = tf.expand_dims(R_jk, 1)
    fc_ij = tf.expand_dims(tf.gather(fc, ind_ij), 1)
    fc_ik = tf.expand_dims(tf.gather(fc, ind_ik), 1)
    fc_jk = tf.expand_dims(fc_jk, 1)
    diff_ij = tf.expand_dims(diff_ij, 1)
    diff_ik = tf.expand_dims(diff_ik, 1)
    eta = tf.expand_dims(eta, 0)
    zeta = tf.expand_dims(zeta, 0)
    lambd = tf.expand_dims(lambd, 0)
    # SF definition
    sf = 2**(1-zeta) *\
        (1+lambd*tf.reduce_sum(diff_ij*diff_ik, axis=-1)/R_ij/R_ik)**zeta *\
        tf.exp(-eta*(R_ij**2+R_ik**2+R_jk**2))*fc_ij*fc_ik*fc_jk
    fp = tf.scatter_nd(tf.expand_dims(i_rind, 1), sf,
                       [tf.reduce_max(a_rind)+1, tf.shape(eta)[1]])
    # Generate Jacobian
    n_sf = sf.shape[-1]
    p_ind, p_uniq_idx = tf.unique(tf.concat([ind_ij, ind_ik], axis=0))
    i_rind = tf.unsorted_segment_max(
        tf.concat([i_rind, i_rind], axis=0), p_uniq_idx, tf.shape(p_ind)[0])
    jacob = tf.stack([tf.gradients(sf[:, i], tensors['diff'])[0]
                      for i in range(n_sf)], axis=2)
    jacob = tf.gather_nd(jacob, tf.expand_dims(p_ind, 1))
    jacob_ind = tf.stack([p_ind, i_rind], axis=1)
    return fp, jacob, jacob_ind


@pi_named('G4_symm_func')
def G4_SF(tensors, lambd, zeta, eta, i="ALL", j="ALL", k="ALL"):
    """BP-style G4 symmetry functions.

    lambd, eta should have the same length,
    each element corresponds to a symmetry function.

    Args:
        lambd: a list of lambda values.
        zeta: a list of zeta values.
        eta: a list of eta values.
        i, j, k: atom types (as int32)

    Returns:
        fp: a (n_atom x n_fingerprint) tensor of fingerprints
            where n_atom is the number of central atoms defined by "i"
        jacob: a (n_pair x n_fingerprint) tensor 
            where n_pair is the number of relavent pairs in this SF
        jacob_ind: a (n_pair) tensor 
            each row correspond to the (p_ind, i_rind) of the pair
            p_ind => the relative position of this pair within all pairs
            i_rind => the index of the central atom for this pair
    """
    if 'ind_3' not in tensors:
        tensors['ind_3'] = _form_triplet(tensors)

    R = tensors['dist']
    fc = tensors['cutoff_func']
    diff = tensors['diff']
    ind_ij = tensors['ind_3'][:, 0]
    ind_ik = tensors['ind_3'][:, 1]
    ind2 = tensors['ind_2']
    i_rind = tf.gather(tensors['ind_2'][:, 0], ind_ij)
    # Build triplet filter
    t_filter = None
    a_rind = tf.cumsum(tf.ones_like(tensors['elems'], tf.int32))-1
    if i != 'ALL':
        i_elem = tf.gather(tensors['elems'], tf.gather(ind2[:, 0], ind_ij))
        t_filter = tf.equal(i_elem, i)
        a_rind = tf.cumsum(tf.cast(tf.equal(tensors['elems'], i), tf.int32))-1
    if j != 'ALL':
        j_elem = tf.gather(tensors['elems'], tf.gather(ind2[:, 1], ind_ij))
        j_filter = tf.equal(j_elem, j)
        t_filter = tf.reduce_all(
            [t_filter, j_filter], axis=0) if t_filter is not None else j_filter
    if k != 'ALL':
        k_elem = tf.gather(tensors['elems'], tf.gather(ind2[:, 1], ind_ik))
        k_filter = tf.equal(k_elem, k)
        t_filter = tf.reduce_all(
            [t_filter, k_filter], axis=0) if t_filter is not None else k_filter
    if t_filter is not None:
        t_ind = tf.where(t_filter)[:, 0]
        ind_ij = tf.gather(ind_ij, t_ind)
        ind_ik = tf.gather(ind_ik, t_ind)
        i_rind = tf.gather(a_rind, tf.gather(i_rind, t_ind))
    # G4 symmetry function
    R_ij = tf.expand_dims(tf.gather(R, ind_ij), 1)
    R_ik = tf.expand_dims(tf.gather(R, ind_ik), 1)
    fc_ij = tf.expand_dims(tf.gather(fc, ind_ij), 1)
    fc_ik = tf.expand_dims(tf.gather(fc, ind_ik), 1)
    diff_ij = tf.expand_dims(tf.gather_nd(diff, tf.expand_dims(ind_ij, 1)), 1)
    diff_ik = tf.expand_dims(tf.gather_nd(diff, tf.expand_dims(ind_ik, 1)), 1)
    eta = tf.expand_dims(eta, 0)
    zeta = tf.expand_dims(zeta, 0)
    lambd = tf.expand_dims(lambd, 0)
    sf = 2**(1-zeta) *\
        (1+lambd*tf.reduce_sum(diff_ij*diff_ik, axis=-1)/R_ij/R_ik)**zeta *\
        tf.exp(-eta*(R_ij**2+R_ik**2))*fc_ij*fc_ik
    fp = tf.scatter_nd(tf.expand_dims(i_rind, 1), sf,
                       [tf.reduce_max(a_rind)+1, tf.shape(eta)[1]])

    # Jacobian generation (perhaps needs some clarification)
    # In short, gradients(sf, diff) gives the non-zero parts of the
    # diff => sf Jacobian (Natom x Natom x 3)

    n_sf = sf.shape[-1]
    p_ind, p_uniq_idx = tf.unique(tf.concat([ind_ij, ind_ik], axis=0))
    i_rind = tf.unsorted_segment_max(
        tf.concat([i_rind, i_rind], axis=0), p_uniq_idx, tf.shape(p_ind)[0])
    jacob = tf.stack([tf.gradients(sf[:, i], tensors['diff'])[0]
                      for i in range(n_sf)], axis=2)
    jacob = tf.gather_nd(jacob, tf.expand_dims(p_ind, 1))
    jacob_ind = tf.stack([p_ind, i_rind], axis=1)
    return fp, jacob, jacob_ind


@pi_named('form_tripet')
def _form_triplet(tensors):
    """Returns triplet indices [ij, jk], where r_ij, r_jk < r_c"""
    p_iind = tensors['ind_2'][:, 0]
    n_atoms = tf.shape(tensors['ind_1'])[0]
    n_pairs = tf.shape(tensors['ind_2'])[0]
    p_aind = tf.cumsum(tf.ones(n_pairs, tf.int32))
    p_rind = p_aind - tf.gather(tf.segment_min(p_aind, p_iind), p_iind)
    t_dense = tf.scatter_nd(tf.stack([p_iind, p_rind], axis=1), p_aind,
                            [n_atoms, tf.reduce_max(p_rind)+1])
    t_dense = tf.gather(t_dense, p_iind)
    t_index = tf.cast(tf.where(t_dense), tf.int32)
    t_ijind = t_index[:, 0]
    t_ikind = tf.gather_nd(t_dense, t_index)-1
    t_ind = tf.gather_nd(tf.stack([t_ijind, t_ikind], axis=1),
                         tf.where(tf.not_equal(t_ijind, t_ikind)))
    return t_ind


@pi_named('bp_symm_func')
def bp_symm_func(tensors, sf_spec, rc, cutoff_type):
    """ Wrapper for building Behler-style symmetry functions"""
    sf_func = {'G2': G2_SF, 'G3': G3_SF, 'G4': G4_SF}
    fps = {}
    for i, sf in enumerate(sf_spec):
        options = {k: v for k, v in sf.items() if k != "type"}
        if sf['type'] == 'G3':  # Workaround for G3 only
            options.update({'rc': rc, 'cutoff_type': cutoff_type})
        fp, jacob, jacob_ind = sf_func[sf['type']](
            tensors,  **options)
        fps['fp_{}'.format(i)] = fp
        fps['jacob_{}'.format(i)] = jacob
        fps['jacob_ind_{}'.format(i)] = jacob_ind
    return fps


@tf.custom_gradient
def _fake_fp(diff, fp, jacob, jacob_ind, n_pairs):
    def _grad(dfp, jacob, jacob_ind):
        # Expand dfp to => (n_pairs x 3 x n_fp)
        dfp = tf.expand_dims(tf.gather_nd(dfp, jacob_ind[:, 1:]), axis=1)
        ddiff = tf.reduce_sum(jacob*dfp, axis=2)
        ddiff = tf.IndexedSlices(ddiff, jacob_ind[:, 0], [n_pairs, 3])
        return ddiff, None, None, None, None
    return tf.identity(fp), lambda dfp: _grad(dfp, jacob, jacob_ind)


def make_fps(tensors, sf_spec, nn_spec, use_jacobian, fp_range, fp_scale):
    fps = {e: [] for e in nn_spec.keys()}
    fps['ALL'] = []
    n_pairs = tf.shape(tensors['diff'])[0]
    for i, sf in enumerate(sf_spec):
        fp = tensors['fp_{}'.format(i)]
        if use_jacobian:
            # connect the diff -> fingerprint gradient
            fp = _fake_fp(tensors['diff'], fp,
                          tensors['jacob_{}'.format(i)],
                          tensors['jacob_ind_{}'.format(i)],
                          n_pairs)
        if fp_scale:
            fp = (fp-fp_range[i][0])/(fp_range[i][1]-fp_range[i][0])*2-1
        fps[sf['i']].append(fp)
    # Deal with "ALL" fingerprints
    fps_all = fps.pop('ALL')
    if fps_all != []:
        fps_all = tf.concat(fps_all, axis=-1)
        for e in nn_spec.keys():
            ind = tf.where(tf.equal(tensors['elems'], e))
            fps[e].append(tf.gather_nd(fps_all, ind))
    # Concatenate all fingerprints
    fps = {k: tf.concat(v, axis=-1) for k, v in fps.items()}
    return fps


def bpnn(tensors, sf_spec, nn_spec,
         rc=5.0, act='tanh', cutoff_type='f1',
         fp_range=[], fp_scale=False,
         preprocess=False, use_jacobian=True):
    """ Network function for Behler-Parrinello Neural Network

    Example of sf_spec::

        [{'type':'G2', 'i': 1, 'j': 8, 'Rs': [1.,2.], 'eta': [0.1,0.2]},
         {'type':'G2', 'i': 8, 'j': 1, 'Rs': [1.,2.], 'eta': [0.1,0.2]},
         {'type':'G4', 'i': 8, 'j': 8, 'lambd':[0.5,1], 'zeta': [1.,2.], 'eta': [0.1,0.2]}]

    The symmetry functions are defined according to the paper:

        Behler, Jörg. “Constructing High-Dimensional Neural Network Potentials: A Tutorial Review.” 
        International Journal of Quantum Chemistry 115, no. 16 (August 15, 2015): 103250. 
        https://doi.org/10.1002/qua.24890.
        (Note the naming of symmetry functiosn are different from http://dx.doi.org/10.1063/1.3553717)

    For more detials about symmetry functions, see the definitions of symmetry functions.

    Example of nn_spec::

        {8: [32, 32, 32], 1: [16, 16, 16]}


    Args:
        tensors: input data (nested tensor from dataset).
        sf_spec (dict): symmetry function specification
        nn_spec (dict): elementwise network specification
            each key points to a list specifying the
            number of nodes in the feed-forward subnets.
        rc (float): cutoff radius.
        cutoff_type (string): cutoff function to use. 
        act (str): activation function to use in dense layers.
        fp_scale (bool): scale the fingeprints according to fp_range.
        fp_range (list of [min, max]): the atomic fingerprint range for each SF
            used to pre-condition the fingerprints.
        preprocess (bool): whether to return the preprocessed tensor.
        use_jacobian (bool): whether to reconnect the grads of fingerprints.
            note that one must use the jacobian if one want forces with preprocess,
            the option is here mainly for verifying the jacobian implementation.

    Returns:
        prediction or preprocessed tensor dictionary
    """
    if 'ind_2' not in tensors:
        tensors.update(cell_list_nl(tensors, rc))
        connect_dist_grad(tensors)
        tensors['cutoff_func'] = cutoff_func(tensors['dist'], cutoff_type, rc)
        tensors.update(bp_symm_func(tensors, sf_spec, rc, cutoff_type))
        if preprocess:
            tensors.pop('dist')
            tensors.pop('cutoff_func')
            tensors.pop('ind_3', None)
            return tensors
    else:
        connect_dist_grad(tensors)
    fps = make_fps(tensors, sf_spec, nn_spec, use_jacobian, fp_range, fp_scale)
    output = 0.0
    n_atoms = tf.shape(tensors['elems'])[0]
    for k, v in nn_spec.items():
        ind = tf.where(tf.equal(tensors['elems'], k))
        with tf.variable_scope("BP_DENSE_{}".format(k)):
            nodes = fps[k]
            for n_node in v:
                nodes = tf.layers.dense(nodes, n_node, activation=act)
            atomic_en = tf.layers.dense(nodes, 1, activation=None,
                                        use_bias=False, name='E_OUT_{}'.format(k))
        output += tf.unsorted_segment_sum(atomic_en[:, 0], ind[:, 0], n_atoms)
    return output

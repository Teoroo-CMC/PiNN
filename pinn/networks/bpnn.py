# -*- coding: utf-8 -*-

import warnings
import tensorflow as tf
from pinn.utils import pi_named, connect_dist_grad
from pinn.layers import CellListNL, CutoffFunc, ANNOutput


@pi_named('G2_symm_func')
def G2_SF(tensors, Rs, eta, i="All", j="ALL"):
    """ BP-style G2 symmetry functions.

    Args:
        i: central atom type, can be "ALL".
        j: neighbor atom type, can be "ALL".
        Rs: a list of Rs values.
        eta: a list of eta values, eta and Rs must have the same length.

    Returns:
        fp: a (n_atom x n_fingerprint) tensor of fingerprints
            where n_atom is the number of central atoms defined by "i"
        jacob_ind: a (n_pair) tensor
            each row correspond to the (p_ind x i_rind) of the pair
            p_ind => the relative position of this pair within all pairs
            i_rind => the index of the central atom for this pair
    """
    R = tensors['dist']
    fc = tensors['fc']
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
    jacob_ind = tf.stack([p_ind, i_rind], axis=1)
    return fp, jacob_ind


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
    R = tensors['dist']
    fc = tensors['fc']
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
    fc_jk = CutoffFunc(cutoff_type=cutoff_type, rc=rc)(R_jk)
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
    i_rind = tf.math.unsorted_segment_max(
        tf.concat([i_rind, i_rind], axis=0), p_uniq_idx, tf.shape(p_ind)[0])
    jacob_ind = tf.stack([p_ind, i_rind], axis=1)
    return fp, jacob_ind


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
        jacob_ind: a (n_pair) tensor
            each row correspond to the (p_ind, i_rind) of the pair
            p_ind => the relative position of this pair within all pairs
            i_rind => the index of the central atom for this pair
    """
    R = tensors['dist']
    fc = tensors['fc']
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
    i_rind = tf.math.unsorted_segment_max(
        tf.concat([i_rind, i_rind], axis=0), p_uniq_idx, tf.shape(p_ind)[0])
    jacob_ind = tf.stack([p_ind, i_rind], axis=1)
    return fp, jacob_ind


@pi_named('form_tripet')
def _form_triplet(tensors):
    """Returns triplet indices [ij, jk], where r_ij, r_jk < r_c"""
    p_iind = tensors['ind_2'][:, 0]
    n_atoms = tf.shape(tensors['ind_1'])[0]
    n_pairs = tf.shape(tensors['ind_2'])[0]
    p_aind = tf.cumsum(tf.ones(n_pairs, tf.int32))
    p_rind = p_aind - tf.gather(tf.math.segment_min(p_aind, p_iind), p_iind)
    t_dense = tf.scatter_nd(tf.stack([p_iind, p_rind], axis=1), p_aind,
                            [n_atoms, tf.reduce_max(p_rind)+1])
    t_dense = tf.gather(t_dense, p_iind)
    t_index = tf.cast(tf.where(t_dense), tf.int32)
    t_ijind = t_index[:, 0]
    t_ikind = tf.gather_nd(t_dense, t_index)-1
    t_ind = tf.gather_nd(tf.stack([t_ijind, t_ikind], axis=1),
                         tf.where(tf.not_equal(t_ijind, t_ikind)))
    return t_ind


bp_sf_fns = {'G2': G2_SF, 'G3': G3_SF, 'G4': G4_SF}
class BPSymmFunc(tf.keras.layers.Layer):
    """ Wrapper for building Behler-style symmetry functions"""
    def __init__(self, sf_spec, rc, cutoff_type, use_jacobian=True):
        super(BPSymmFunc, self).__init__()
        # specifications
        self.sf_spec = []
        self.triplet = False
        for spec in sf_spec:
            fn = bp_sf_fns[spec['type']]
            options = {k:v for k,v in spec.items() if k!='type'}
            if spec['type'] == 'G3':
                options.update({'rc': rc, 'cutoff_type': cutoff_type})
            if spec['type'] in ['G3', 'G4']:
                self.triplet = True
            self.sf_spec.append((fn, options))
        self.use_jacobian = use_jacobian
        # the cutoff function needs to be called from here to
        # calculation the jacobian
        self.fc_layer = CutoffFunc(rc, cutoff_type)

    def call(self, tensors):
        if self.triplet:
            tensors['ind_3'] = _form_triplet(tensors)

        fps = {}
        if self.use_jacobian:
            with tf.GradientTape(persistent=True) as gtape:
                gtape.watch(tensors['diff'])
                connect_dist_grad(tensors)
                tensors['fc'] = self.fc_layer(tensors['dist'])
                for i, (fn, options) in enumerate(self.sf_spec):
                    fp, jacob_ind = fn(tensors, **options)
                    fps['fp_{}'.format(i)] = fp
                    fp_slices = [fp[:,j] for j in range(fp.shape[1])]
                    with gtape.stop_recording():
                        warnings.filterwarnings('ignore')
                        jacob = tf.stack([
                            gtape.gradient(fp_slice, tensors['diff'])
                            for fp_slice in fp_slices], axis=2)
                        warnings.resetwarnings()
                        jacob = tf.gather_nd(jacob, jacob_ind[:,:1])
                        fps['jacob_{}'.format(i)] = jacob
                        fps['jacob_ind_{}'.format(i)] = jacob_ind
        else:
            tensors['fc'] = self.fc_layer(tensors['dist'])
            for i, (fn, options) in enumerate(self.sf_spec):
                fp, jacob_ind = fn(tensors, **options)
                fps['fp_{}'.format(i)] = fp
        tensors.update(fps)
        tensors.pop('ind_3')
        tensors.pop('dist')
        tensors.pop('fc')
        return tensors


@tf.custom_gradient
def _fake_fp(diff, fp, jacob, jacob_ind, n_pairs):
    def _grad(dfp, jacob, jacob_ind):
        # Expand dfp to => (n_pairs x 3 x n_fp)
        dfp = tf.expand_dims(tf.gather_nd(dfp, jacob_ind[:, 1:]), axis=1)
        ddiff = tf.reduce_sum(jacob*dfp, axis=2)
        ddiff = tf.IndexedSlices(ddiff, jacob_ind[:, 0], [n_pairs, 3])
        return ddiff, None, None, None, None
    return tf.identity(fp), lambda dfp: _grad(dfp, jacob, jacob_ind)


class BPFingerprint(tf.keras.layers.Layer):
    """Layer for make BP style Atomic Fingerprints"""
    def __init__(self, sf_spec, nn_spec, fp_range, fp_scale, use_jacobian=True):
        super(BPFingerprint, self).__init__()
        self.sf_spec = sf_spec
        self.nn_spec = nn_spec
        self.fp_range = fp_range
        self.fp_scale = fp_scale
        self.use_jacobian = use_jacobian

    def call(self, tensors):
        sf_spec, nn_spec, fp_range, fp_scale, use_jacobian = self.sf_spec, self.nn_spec, self.fp_range, self.fp_scale, self.use_jacobian
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
        tensors['elem_fps'] = fps
        return tensors


class BPFeedForward(tf.keras.layers.Layer):
    """Element specific feed-forward neural networks used in BPNN"""
    def __init__(self, nn_spec, act, out_units):
        super(BPFeedForward, self).__init__()
        self.ff_layers = {}
        self.out_units = out_units
        for k, v in nn_spec.items():
            self.ff_layers[k] = [
                tf.keras.layers.Dense(units, activation=act)
                for units in v]
            self.ff_layers[k].append(
                tf.keras.layers.Dense(out_units, activation=None,
                                      use_bias=False))

    def call(self, tensors):
        output = []
        indices = []
        for k, layers in self.ff_layers.items():
            tensor_elem = tensors['elem_fps'][k]
            for layer in layers:
                tensor_elem = layer(tensor_elem)
            output.append(tensor_elem)
            indices.append(tf.where(tf.equal(tensors['elems'], k))[:,0])

        output = tf.math.unsorted_segment_sum(
            tf.concat(output, axis=0),
            tf.concat(indices, axis=0),
            tf.shape(tensors['ind_1'])[0])

        return output

class PreprocessLayer(tf.keras.layers.Layer):
    def __init__(self, sf_spec, rc, cutoff_type, use_jacobian):
        super(PreprocessLayer, self).__init__()
        self.nl_layer = CellListNL(rc)
        self.symm_func = BPSymmFunc(sf_spec, rc, cutoff_type, use_jacobian)

    def call(self, tensors):
        tenors = tensors.copy()
        for k in ['elems', 'dist']:
            if k in tensors.keys():
                tensors[k] = tf.reshape(tensors[k], tf.shape(tensors[k])[:1])
        if 'ind_2' not in tensors:
            tensors.update(self.nl_layer(tensors))
            tensors = self.symm_func(tensors)
        return tensors


class BPNN(tf.keras.Model):
    """ Network function for Behler-Parrinello Neural Network

    Example of sf_spec::

        [{'type':'G2', 'i': 1, 'j': 8, 'Rs': [1.,2.], 'eta': [0.1,0.2]},
         {'type':'G2', 'i': 8, 'j': 1, 'Rs': [1.,2.], 'eta': [0.1,0.2]},
         {'type':'G4', 'i': 8, 'j': 8, 'lambd':[0.5,1], 'zeta': [1.,2.], 'eta': [0.1,0.2]}]

    The symmetry functions are defined according to the paper:

        Behler, Jörg. “Constructing High-Dimensional Neural Network Potentials: A Tutorial Review.”
        International Journal of Quantum Chemistry 115, no. 16 (August 15, 2015): 103250.
        https://doi.org/10.1002/qua.24890.
        (Note the naming of symmetry functions is different from http://dx.doi.org/10.1063/1.3553717)

    For more detials about symmetry functions, see the definitions of symmetry functions.

    Example of nn_spec::

        {8: [32, 32, 32], 1: [16, 16, 16]}


    Args:
        tensors: input data (nested tensor from dataset).
        sf_spec (dict): symmetry function specification.
        nn_spec (dict): elementwise network specification,
            each key points to a list specifying the
            number of nodes in the feed-forward subnets.
        rc (float): cutoff radius.
        cutoff_type (string): cutoff function to use.
        act (str): activation function to use in dense layers.
        fp_scale (bool): scale the fingerprints according to fp_range.
        fp_range (list of [min, max]): the atomic fingerprint range for each SF
            used to pre-condition the fingerprints.
        use_jacobian (bool): whether to reconnect the grads of fingerprints.
            note that one must use the jacobian if one want forces with
            preprocessing, the option is here mainly for verifying the
            jacobian implementation.

    Returns:
        prediction or preprocessed tensor dictionary
    """
    def __init__(self, sf_spec, nn_spec,
                 rc=5.0, act='tanh', cutoff_type='f1',
                 fp_range=[], fp_scale=False,
                 preprocess=False, use_jacobian=True,
                 out_units=1, out_pool=False):
        super(BPNN, self).__init__()
        self.preprocess = PreprocessLayer(sf_spec, rc, cutoff_type, use_jacobian)
        self.fingerprint = BPFingerprint(sf_spec, nn_spec, fp_range, fp_scale, use_jacobian)
        self.feed_forward = BPFeedForward(nn_spec, act, out_units)
        self.ann_output = ANNOutput(out_pool)

    def call(self, tensors):
        tensors = self.preprocess(tensors)
        tensors = self.fingerprint(tensors)
        output = self.feed_forward(tensors)
        output = self.ann_output([tensors['ind_1'], output])
        return output

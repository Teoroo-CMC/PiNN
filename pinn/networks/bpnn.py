# -*- coding: utf-8 -*-

import warnings
import tensorflow as tf
from pinn.utils import pi_named, connect_dist_grad
from pinn.layers.bpsf import G2_SF, G3_SF, G4_SF
from pinn.layers import CellListNL, CutoffFunc, ANNOutput


@tf.custom_gradient
def _fake_fp(diff, fp, jacob, jacob_ind, n_pairs):
    def _grad(dfp, jacob, jacob_ind):
        # Expand dfp to => (n_pairs x 3 x n_fp)
        dfp = tf.expand_dims(tf.gather_nd(dfp, jacob_ind[:, 1:]), axis=1)
        ddiff = tf.reduce_sum(jacob*dfp, axis=2)
        ddiff = tf.IndexedSlices(ddiff, jacob_ind[:, 0], [n_pairs, 3])
        return ddiff, None, None, None, None
    return tf.identity(fp), lambda dfp: _grad(dfp, jacob, jacob_ind)


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


class BPSymmFunc(tf.keras.layers.Layer):
    """ Wrapper for building Behler-style symmetry functions"""
    def __init__(self, sf_spec, rc, cutoff_type, use_jacobian=True):
        super(BPSymmFunc, self).__init__()
        bpsf_layers = {'G2': G2_SF, 'G3': G3_SF, 'G4': G4_SF}
        # specifications
        self.bpsfs = []
        self.triplet = False
        self.fc_layer = CutoffFunc(rc, cutoff_type)

        for spec in sf_spec:
            layer = bpsf_layers[spec['type']]
            args = {k:v for k,v in spec.items() if k!='type'}
            if spec['type'] == 'G3':
                args.update({'cutoff': self.fc_layer, 'rc': rc})
            if spec['type'] in ['G3', 'G4']:
                self.triplet = True
            self.bpsfs.append(layer(**args))
        self.use_jacobian = use_jacobian

    def _compute_fps(self, tensors, gtape=None):
        fc = self.fc_layer(tensors['dist'])
        fps = {}
        for i, layer in enumerate(self.bpsfs):
            if isinstance(layer, G2_SF):
                fp, jacob_ind = layer(
                    tensors["ind_2"],
                    dist=tensors["dist"],
                    elems=tensors["elems"],
                    fc=fc,
                )
            else:
                fp, jacob_ind = layer(
                    tensors["ind_2"],
                    ind_3=tensors["ind_3"],
                    dist=tensors["dist"],
                    diff=tensors["diff"],
                    elems=tensors["elems"],
                    fc=fc,
                )
            fps[f'fp_{i}'] = fp

            # compute jacobian when a gradient tape is provided
            if gtape is not None:
                fp_slices = [fp[:,j] for j in range(fp.shape[1])]
                with gtape.stop_recording():
                    warnings.filterwarnings('ignore')
                    jacob = tf.stack([
                        gtape.gradient(fp_slice, tensors['diff'])
                        for fp_slice in fp_slices], axis=2)
                    warnings.resetwarnings()
                    jacob = tf.gather_nd(jacob, jacob_ind[:,:1])
                fps[f'jacob_{i}'] = jacob
                fps[f'jacob_ind_{i}'] = jacob_ind
        return fps

    def call(self, tensors):
        if self.triplet:
            tensors['ind_3'] = _form_triplet(tensors)
        fps = {}
        if self.use_jacobian:
            with tf.GradientTape(persistent=True) as gtape:
                gtape.watch(tensors['diff'])
                connect_dist_grad(tensors)
                fps = self._compute_fps(tensors, gtape = gtape)
        else:
            fps = self._compute_fps(tensors)
        tensors.update(fps)
        tensors.pop('dist')
        if self.triplet:
            tensors.pop('ind_3')
        return tensors


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
                fp_min = tf.expand_dims(tf.cast(fp_range[i][0], fp.dtype), axis=0)
                fp_max = tf.expand_dims(tf.cast(fp_range[i][1], fp.dtype), axis=0)
                fp = (fp-fp_min)/tf.maximum(fp_max-fp_min, 1e-6)*2-1
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
                for i, units in enumerate(v)]
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

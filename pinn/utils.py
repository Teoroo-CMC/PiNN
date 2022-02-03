# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from functools import wraps


def init_params(params, dataset):
    """Initlaize the parameters with a dataset

    For potential models, generate the atomic dress from the dataset.

    For BPNN, generate the range of fingerprints (to be used with `fp_scale`)

    Args:
       params (dict): the parameter dictionary
       dataset (dataset): a tensorflow dataset

    """
    if params['model']['name']=='potential_model':
        if 'e_dress' not in params['model']['params']:
            elems = []
            for e in dataset.map(lambda x: x['elems']).as_numpy_iterator():
                elems = np.unique(np.concatenate([elems,e]))
        else:
            elems = list(params['model']['params']['e_dress'].keys())
        elems = [int(e) for e in elems]
        print('Fitting an atomic dress from the training set.')
        e_dress, err = get_atomic_dress(dataset, elems)
        print(f' RMSE after substracting the dress: {np.sqrt(np.mean(err**2)):.6e}')
        params['model']['params']['e_dress'] = e_dress

    if params['network']['name']=='BPNN'\
       and 'fp_scale' in params['network']['params']\
       and params['network']['params']['fp_scale']:
        print('Generating the fp range from the training set (will take a while).')
        fp_range = get_fp_range(params, dataset)
        params['network']['params']['fp_range'] = fp_range


def get_fp_range(params, dataset):
    """Generate the range of fingerprints for BPNN

    Args:
       params (dict): the parameter dictionary
       dataset (dataset): a tensorflow dataset

    Returns
       a list of ranges, one for each fp specification
    """
    import sys, pinn, copy
    from pinn.io import sparse_batch
    if 'ind_1' not in next(iter(dataset)):
        dataset = dataset.apply(sparse_batch(1))
    network = pinn.get_network(copy.deepcopy(params['network']))
    dataset = dataset.map(network.preprocess).as_numpy_iterator()
    fp_range = {int(k[3:]): [np.min(v, initial=np.Inf, axis=0),
                             np.max(v, initial=0, axis=0)]
                for k,v in next(dataset).items() if k.startswith('fp')}
    for i, tensors in enumerate(dataset):
        sys.stdout.write(f'\r {i+1} samples scanned for fp_range ...')
        for k, v in fp_range.items():
            stacked0 = np.concatenate([[v[0]], tensors[f'fp_{k}']],axis=0)
            stacked1 = np.concatenate([[v[1]], tensors[f'fp_{k}']],axis=0)
            v[0] = np.min(stacked0, axis=0).tolist()
            v[1] = np.max(stacked1, axis=0).tolist()
    fp_range = [fp_range[i] for i in range(len(fp_range.keys()))]
    print(f'\r {i+1} samples scanned for fp_range, done.')
    return fp_range


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
    e_dress = tf.math.unsorted_segment_sum(
        e_dress, tensors['ind_1'][:, 0], n_batch)
    return e_dress


def count_atoms(ind_1, dtype):
    return tf.math.unsorted_segment_sum(
        tf.ones_like(ind_1, dtype), ind_1, tf.reduce_max(ind_1)+1)


def get_atomic_dress(dataset, elems, key='e_data'):
    """Fit the atomic energy with a element dependent atomic dress

    Args:
        dataset: dataset to fit
        elems: a list of element numbers
        key: key of the property to fit

    Returns:
        atomic_dress: a dictionary comprising the atomic energy of each element
        error: residue error of the atomic dress
    """
    from pinn.io import sparse_batch

    def count_elems(tensors):
        tensors = tensors.copy()
        count = tf.equal(tf.expand_dims(
            tensors['elems'], 1), tf.expand_dims(elems, 0))
        count = tf.cast(count, tf.int32)
        count = tf.math.segment_sum(count, tensors['ind_1'][:, 0])
        return count, tensors[key]

    if 'ind_1' not in next(iter(dataset)):
        dataset = dataset.apply(sparse_batch(1))
    x, y = [], []
    for x_i, y_i in dataset.map(count_elems).as_numpy_iterator():
        x.append(x_i)
        y.append(y_i)

    x, y = np.concatenate(x, 0), np.concatenate(y, 0)
    beta = np.dot(np.dot(np.linalg.pinv(np.dot(x.T, x)), x.T), np.array(y))
    dress = {e: float(beta[i]) for (i, e) in enumerate(elems)}
    error = np.dot(x, beta) - y
    return dress, error


def pi_named(default_name='unnamed'):
    """Decorate a layer to have a name"""
    def decorator(func):
        @wraps(func)
        def named_layer(*args, name=default_name, **kwargs):
            with tf.name_scope(name):
                return func(*args, **kwargs)
        return named_layer
    return decorator


def TuneTrainable(train_fn):
    """Helper function for geting a trainable to use with Tune

    The function expectes a train_fn function which takes a config as input,
    and returns four items.

    - model: the tensorflow estimator.
    - train_spec: training specification.
    - eval_spec: evaluation specification.
    - reporter: a function which returns the metrics given evalution.

    The resulting trainable reports the metrics when checkpoints are saved,
    the report frequency is controlled by the checkpoint frequency,
    and the metrics are determined by reporter.
    """
    import os
    from ray.tune import Trainable
    from tensorflow.train import CheckpointSaverListener

    class _tuneStoper(CheckpointSaverListener):
        def after_save(self, session, global_step_value):
            return True

    class TuneTrainable(Trainable):
        def _setup(self, config):
            tf.logging.set_verbosity(tf.logging.ERROR)
            self.config = config
            model, train_spec, eval_spec, reporter = train_fn(config)
            self.model = model
            self.train_spec = train_spec
            self.eval_spec = eval_spec
            self.reporter = reporter

        def _train(self):
            import warnings
            index_warning = 'Converting sparse IndexedSlices'
            warnings.filterwarnings('ignore', index_warning)
            model = self.model
            model.train(input_fn=self.train_spec.input_fn,
                        max_steps=self.train_spec.max_steps,
                        hooks=self.train_spec.hooks,
                        saving_listeners=[_tuneStoper()])
            eval_out = model.evaluate(input_fn=self.eval_spec.input_fn,
                                      steps=self.eval_spec.steps,
                                      hooks=self.eval_spec.hooks)
            metrics = self.reporter(eval_out)
            return metrics

        def _save(self, checkpoint_dir):
            latest_checkpoint = self.model.latest_checkpoint()
            chkpath = os.path.join(checkpoint_dir, 'path.txt')
            with open(chkpath, 'w') as f:
                f.write(latest_checkpoint)
            return chkpath

        def _restore(self, checkpoint_path):
            with open(checkpoint_path) as f:
                chkpath = f.readline().strip()
            self.model, _, _, _ = train_fn(self.config, chkpath)
    return TuneTrainable


def connect_dist_grad(tensors):
    """This function assumes tensors is a dictionary containing 'ind_2',
    'diff' and 'dist' from a neighbor list layer It rewirtes the
    'dist' and 'dist' tensor so that their gradients are properly
    propogated during force calculations
    """
    tensors['diff'] = _connect_diff_grad(tensors['coord'], tensors['diff'],
                                         tensors['ind_2'])
    if 'dist' in tensors:
        # dist can be deleted if the jacobian is cached, so we may skip this
        tensors['dist'] = _connect_dist_grad(tensors['diff'], tensors['dist'])


@tf.custom_gradient
def _connect_diff_grad(coord, diff, ind):
    """Returns a new diff with its gradients connected to coord"""
    def _grad(ddiff, coord, diff, ind):
        natoms = tf.shape(coord)[0]
        if type(ddiff) == tf.IndexedSlices:
            # handle sparse gradient inputs
            ind = tf.gather_nd(ind, tf.expand_dims(ddiff.indices, 1))
            ddiff = ddiff.values
        dcoord = tf.math.unsorted_segment_sum(ddiff, ind[:, 1], natoms)
        dcoord -= tf.math.unsorted_segment_sum(ddiff, ind[:, 0], natoms)
        return dcoord, None, None
    return tf.identity(diff), lambda ddiff: _grad(ddiff, coord, diff, ind)


@tf.custom_gradient
def _connect_dist_grad(diff, dist):
    """Returns a new dist with its gradients connected to diff"""
    def _grad(ddist, diff, dist):
        return tf.expand_dims(ddist/dist, 1)*diff, None
    return tf.identity(dist), lambda ddist: _grad(ddist, diff, dist)


@pi_named('form_basis_jacob')
def make_basis_jacob(basis, diff):
    jacob = [tf.gradients(basis[:, i], diff)[0]
             for i in range(basis.shape[1])]
    jacob = tf.stack(jacob, axis=2)
    return jacob


def connect_basis_jacob(tensors):
    tensors['basis'] = _connect_basis_grad(
        tensors['diff'], tensors['basis'], tensors['jacob'])


@tf.custom_gradient
def _connect_basis_grad(diff, basis, jacob):
    def _grad(dbasis, jacob):
        ddiff = jacob * tf.expand_dims(dbasis, 1)
        ddiff = tf.reduce_sum(ddiff, axis=2)
        return ddiff, None, None
    return tf.identity(basis), lambda dbasis: _grad(dbasis, jacob)

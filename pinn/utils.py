# -*- coding: utf-8 -*-
""" Misc tools"""

import tensorflow as tf
import numpy as np
from functools import wraps


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
    def count_elems(tensors):
        if 'ind_1' not in tensors:
            tensors['ind_1'] = tf.expand_dims(tf.zeros_like(tensors['elems']), 1)
            tensors[key] = tf.expand_dims(tensors[key], 0)
        count = tf.equal(tf.expand_dims(
            tensors['elems'], 1), tf.expand_dims(elems, 0))
        count = tf.cast(count, tf.int32)
        count = tf.math.segment_sum(count, tensors['ind_1'][:, 0])
        return count, tensors[key]

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

# -*- coding: utf-8 -*-
"""Models interfaces neural networks with the estimator api
Models transfer a network to a model function to use with the tf estimator api.
A model defines the goal/loss of the model, as well as training paramters.
"""
from pinn.networks import pinet
import pinn.networks
import tensorflow as tf
import numpy as np

from pinn.utils import pi_named

default_params = {
    ### Scaling and units
    # The loss function will be MSE((pred - label) * scale)
    # For vector/tensor predictions
    # the error will be pre-component instead of per-atom
    # d_unit is the unit of dipole to report w.r.t the input labels
    'd_scale': 1.0,  # dipole scale for prediction
    'd_unit': 1.0,  # output unit of dipole during prediction
    # Loss function options
    'max_dipole': False,     # if set to float, omit dipoles larger than it
    'use_d_per_atom': False,  # use d_per_atom to calculate d_loss
    'use_d_per_sqrt': False,
    'log_d_per_atom': False,  # log d_per_atom and its distribution
                             # ^- this is forcely done if use_d_per_atom
    'use_d_weight': False,   # scales the loss according to d_weight
    'use_l2': False,         # L2 regularization
    # Loss function multipliers
    'd_loss_multiplier': 1.0,
    'l2_loss_multiplier': 1.0,
    # Optimizer related
    'learning_rate': 3e-4,   # Learning rate
    'use_norm_clip': True,   # see tf.clip_by_global_norm
    'norm_clip': 0.01,       # see tf.clip_by_global_norm
    'use_decay': True,       # Exponential decay
    'decay_step': 10000,      # every ? steps
    'decay_rate': 0.999,      # scale by ?
}


def dipole_model(params, **kwargs):
    """Shortcut for generating dipole model from paramters
    When creating the model, a params.yml is automatically created 
    in model_dir containing network_params and model_params.
    The dipole model can also be initiated with the model_dir, 
    in that case, params.yml must locate in model_dir from which
    all parameters are loaded
    Args:
        params(str or dict): parameter dictionary or the model_dir
        **kwargs: additional options for the estimator, e.g. config
    """
    import os
    import yaml
    from datetime import datetime

    if isinstance(params, str):
        model_dir = params
        assert tf.gfile.Exists('{}/params.yml'.format(model_dir)),\
            "Parameters files not found."
        with FileIO(os.path.join(model_dir, 'params.yml'), 'r') as f:
            params = yaml.load(f, Loader=yaml.Loader)
    else:
        model_dir = params['model_dir']
        yaml.Dumper.ignore_aliases = lambda *args: True
        to_write = yaml.dump(params)
        params_path = os.path.join(model_dir, 'params.yml')
        if not tf.gfile.IsDirectory(model_dir):
            tf.gfile.MakeDirs(model_dir)
        if tf.gfile.Exists(params_path):
            original = FileIO(params_path, 'r').read()
            if original != to_write:
                tf.gfile.Rename(params_path, params_path+'.' +
                          datetime.now().strftime('%y%m%d%H%M'))
        FileIO(params_path, 'w').write(to_write)

    model = tf.estimator.Estimator(
        model_fn=_dipole_model_fn, params=params,
        model_dir=model_dir, **kwargs)
    return model


def _dipole_model_fn(features, labels, mode, params):
    """Model function for neural network dipoles"""
    if isinstance(params['network'], str):
        network_fn = getattr(pinn.networks, params['network'])
    else:
        network_fn = params['network']

    network_params = params['network_params']
    model_params = default_params.copy()
    model_params.update(params['model_params'])
    pred = network_fn(features, **network_params)
    pred = tf.expand_dims(pred, axis=1)

    ind = features['ind_1']  # ind_1 => id of molecule for each atom
    nbatch = tf.reduce_max(ind)+1
    charge = tf.unsorted_segment_sum(pred, ind[:, 0], nbatch)

    dipole = pred * features['coord']
    dipole = tf.unsorted_segment_sum(dipole, ind[:, 0], nbatch)
    dipole = tf.sqrt(tf.reduce_sum(dipole**2, axis=1)+1e-6)
    #charge = charge[:,0]

    if mode == tf.estimator.ModeKeys.TRAIN:
        n_trainable = np.sum([np.prod(v.shape)
                              for v in tf.trainable_variables()])
        print("Total number of trainable variables: {}".format(n_trainable))

        loss, metrics = _get_loss(features, dipole, charge, model_params)
        _make_train_summary(metrics)
        train_op = _get_train_op(loss,  model_params)
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        loss, metrics = _get_loss(features, dipole, charge, model_params)
        metrics = _make_eval_metrics(metrics)
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    if mode == tf.estimator.ModeKeys.PREDICT:
        pred = pred / model_params['d_scale']
        pred *= model_params['d_unit']

        predictions = {
            'dipole': dipole,
            'charges': tf.expand_dims(pred, 0)
        }
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions)


def _get_dense_grad(dipole, coord):
    """get a gradient and convert to dense form"""
    import warnings
    index_warning = 'Converting sparse IndexedSlices'
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', index_warning)
        grad = tf.gradients(dipole, coord)[0]
    if type(grad) == tf.IndexedSlices:
        grad = tf.scatter_nd(tf.expand_dims(grad.indices, 1), grad.values,
                             tf.cast(grad.dense_shape, tf.int32))
    return grad


@pi_named('LOSSES')
def _get_loss(features, dipole, charge, model_params):
    metrics = {}  # Not editting features here for safety, use a separate dict

    d_pred = dipole
    d_data = features['d_data']
    d_data *= model_params['d_scale']
    if model_params['max_dipole']:
        # should get the mask here since max_dipole refers to total dipole
        d_mask = tf.abs(d_data) > model_params['max_dipole']

    d_error = dipole - d_data
    metrics['d_data'] = d_data
    metrics['d_pred'] = d_pred
    metrics['d_error'] = d_error
    metrics['q_data'] = tf.zeros_like(charge)
    metrics['q_pred'] = charge
    metrics['q_error'] = charge

    if model_params['log_d_per_atom'] or model_params['use_d_per_atom']:
        ind_1 = features['ind_1']
        atom_count = tf.unsorted_segment_sum(
            tf.ones_like(ind_1, tf.float32), ind_1, tf.shape(d_data)[0])
        d_pred_per_atom = d_pred/atom_count
        d_data_per_atom = d_data/atom_count
        d_error_per_atom = d_error/atom_count
        metrics['d_data_per_atom'] = d_data_per_atom
        metrics['d_pred_per_atom'] = d_pred_per_atom
        metrics['d_error_per_atom'] = d_error_per_atom

    # e_error is ajusted from here
    if model_params['use_d_per_atom']:
        d_error = d_error_per_atom
        if model_params['use_d_per_sqrt']:
            d_error = d_error_per_atom*tf.sqrt(atom_count)
    if model_params['use_d_weight']:
        # Add this to metrics so that one can get a weighed RMSE
        metrics['d_weight'] = features['d_weight']
        d_error *= features['d_weight']
    if model_params['max_dipole']:
        d_error = tf.where(d_mask, tf.zeros_like(d_error), d_error)
    # keep the per_sample loss so that it can be consumed by tf.metrics.mean
    d_loss = d_error**2 * model_params['d_loss_multiplier']
    q_loss = charge**2
    metrics['d_loss'] = d_loss
    tot_loss = tf.reduce_mean(d_loss) + tf.reduce_mean(q_loss)

    if model_params['use_l2']:
        tvars = tf.trainable_variables()
        l2_loss = tf.add_n([
            tf.nn.l2_loss(v) for v in tvars if
            ('bias' not in v.name and 'D_OUT' not in v.name)])
        metrics['l2_loss'] = l2_loss * model_params['l2_loss_multiplier']
        tot_loss += l2_loss

    metrics['tot_loss'] = tot_loss
    return tot_loss, metrics


@pi_named('METRICS')
def _make_eval_metrics(metrics):
    eval_metrics = {
        'METRICS/D_MAE': tf.metrics.mean_absolute_error(
            metrics['d_data'], metrics['d_pred']),
        'METRICS/D_RMSE': tf.metrics.root_mean_squared_error(
            metrics['d_data'], metrics['d_pred']),
        'METRICS/D_LOSS': tf.metrics.mean(metrics['d_loss']),
        'METRICS/Q_MAE': tf.metrics.mean_absolute_error(
            metrics['q_data'], metrics['q_pred']),
        'METRICS/Q_RMSE': tf.metrics.root_mean_squared_error(
            metrics['q_data'], metrics['q_pred']),
        'METRICS/TOT_LOSS': tf.metrics.mean(metrics['tot_loss'])
    }

    if 'd_data_per_atom' in metrics:
        eval_metrics['METRICS/D_PER_ATOM_MAE'] = tf.metrics.mean_absolute_error(
            metrics['d_data_per_atom'], metrics['d_pred_per_atom'])
        eval_metrics['METRICS/D_PER_ATOM_RMSE'] = tf.metrics.root_mean_squared_error(
            metrics['d_data_per_atom'], metrics['d_pred_per_atom'])

    if 'l2_loss' in metrics:
        eval_metrics['METRICS/L2_LOSS'] = tf.metrics.mean(metrics['l2_loss'])
    return eval_metrics


@pi_named('METRICS')
def _make_train_summary(metrics):
    tf.summary.scalar('D_RMSE', tf.sqrt(tf.reduce_mean(metrics['d_error']**2)))
    tf.summary.scalar('D_MAE', tf.reduce_mean(tf.abs(metrics['d_error'])))
    tf.summary.scalar('D_LOSS', tf.reduce_mean(metrics['d_loss']))
    tf.summary.scalar('Q_RMSE', tf.sqrt(tf.reduce_mean(metrics['q_error']**2)))
    tf.summary.scalar('Q_MAE', tf.reduce_mean(tf.abs(metrics['q_error'])))
    tf.summary.scalar('TOT_LOSS', metrics['tot_loss'])
    tf.summary.histogram('D_DATA', metrics['d_data'])
    tf.summary.histogram('D_PRED', metrics['d_pred'])
    tf.summary.histogram('D_ERROR', metrics['d_error'])

    if 'd_data_per_atom' in metrics:
        tf.summary.scalar(
            'D_PER_ATOM_MAE',
            tf.reduce_mean(tf.abs(metrics['d_error_per_atom'])))
        tf.summary.scalar(
            'D_PER_ATOM_RMSE',
            tf.sqrt(tf.reduce_mean(metrics['d_error_per_atom']**2)))
        tf.summary.histogram('D_PER_ATOM_DATA', metrics['d_data_per_atom'])
        tf.summary.histogram('D_PER_ATOM_PRED', metrics['d_pred_per_atom'])
        tf.summary.histogram('D_PER_ATOM_ERROR', metrics['d_error_per_atom'])

    if 'l2_loss' in metrics:
        tf.summary.scalar('L2_LOSS', metrics['l2_loss'])


@pi_named('TRAIN_OP')
def _get_train_op(loss, model_params):
    # Get the optimizer
    global_step = tf.train.get_global_step()
    learning_rate = model_params['learning_rate']
    if model_params['use_decay']:
        learning_rate = tf.train.exponential_decay(
            learning_rate, global_step,
            model_params['decay_step'], model_params['decay_rate'],
            staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # Get the gradients
    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)
    if model_params['use_norm_clip']:
        grads, _ = tf.clip_by_global_norm(grads, model_params['norm_clip'])
    return optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

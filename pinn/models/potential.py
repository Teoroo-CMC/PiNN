# -*- coding: utf-8 -*-
"""Models interfaces neural networks with the estimator api

Models transfer a network to a model function to use with the tf estimator api.
A model defines the goal/loss of the model, as well as training paramters.
"""
import pinn.networks
import tensorflow as tf
import numpy as np

from pinn.layers import atomic_dress
from pinn.utils import pi_named, connect_dist_grad

default_params = {
    ### Scaling and units
    # The loss function will be MSE((pred - label) * scale)
    # For vector/tensor predictions
    # the error will be pre-component instead of per-atom
    # e_unit is the unit of energy to report w.r.t the input labels
    # no f_unit yet, f_unit is just e_unit/input coordinate unit
    # e.g. if one have input in Hartree, scales it by 100 for training
    #      and output eV when report error
    #      then e_scale should be 100, and e_unit = hartree2evp
    'e_dress': {},  # element-specific energy dress
    'e_scale': 1.0,  # energy scale for prediction
    'e_unit': 1.0,  # output unit of energy during prediction
    # Loss function options
    ## Energy
    'max_energy': False,     # if set to float, omit energies larger than it
    'use_e_per_atom': False,  # use e_per_atom to calculate e_loss
    'use_e_per_sqrt': False,
    'log_e_per_atom': False,  # log e_per_atom and its distribution
                             # ^- this is forcely done if use_e_per_atom
    'use_e_weight': False,   # scales the loss according to e_weight
    ## Force
    'use_force': False,      # include force in Loss function
    'max_force': False,      # if set to float, omit forces larger than it
    'use_f_weights': False,  # scales the loss according to f_weights
    ## Stress
    'use_stress': False,      # include stress in Loss function
    ## L2
    'use_l2': False,         # L2 regularization
    # Loss function multipliers
    'e_loss_multiplier': 1.0,
    'f_loss_multiplier': 1.0,
    's_loss_multiplier': 1.0,
    'l2_loss_multiplier': 1.0,
    # Optimizer related
    'learning_rate': 3e-4,   # Learning rate
    'use_norm_clip': True,   # see tf.clip_by_global_norm
    'norm_clip': 0.01,       # see tf.clip_by_global_norm
    'use_decay': True,       # Exponential decay
    'decay_step': 10000,      # every ? steps
    'decay_rate': 0.999,      # scale by ?
}


def potential_model(params, **kwargs):
    """Shortcut for generating potential model from paramters

    When creating the model, a params.yml is automatically created 
    in model_dir containing network_params and model_params.

    The potential model can also be initiated with the model_dir, 
    in that case, params.yml must locate in model_dir from which
    all parameters are loaded

    Args:
        params(str or dict): parameter dictionary or the model_dir
        **kwargs: additional options for the estimator, e.g. config
    """
    import os
    import yaml
    from tensorflow.python.lib.io.file_io import FileIO
    from datetime import datetime

    if isinstance(params, str):
        model_dir = params
        assert tf.io.gfile.exists('{}/params.yml'.format(model_dir)),\
            "Parameters files not found."
        with FileIO(os.path.join(model_dir, 'params.yml'), 'r') as f:
            params = yaml.load(f, Loader=yaml.Loader)
    else:
        model_dir = params['model_dir']
        yaml.Dumper.ignore_aliases = lambda *args: True
        to_write = yaml.dump(params)
        params_path = os.path.join(model_dir, 'params.yml')
        if not tf.io.gfile.isdir(model_dir):
            tf.io.gfile.makedirs(model_dir)
        if tf.io.gfile.exists(params_path):
            original = FileIO(params_path, 'r').read()
            if original != to_write:
                tf.gfile.Rename(params_path, params_path+'.' +
                          datetime.now().strftime('%y%m%d%H%M'))
        FileIO(params_path, 'w').write(to_write)

    model = tf.estimator.Estimator(
        model_fn=_potential_model_fn, params=params,
        model_dir=model_dir, **kwargs)
    return model


def _potential_model_fn(features, labels, mode, params):
    """Model function for neural network potentials"""
    if isinstance(params['network'], str):
        network_fn = getattr(pinn.networks, params['network'])
    else:
        network_fn = params['network']

    network_params = params['network_params']
    model_params = default_params.copy()
    model_params.update(params['model_params'])

    network = network_fn(**network_params)
    features = network.preprocess(features)
    connect_dist_grad(features)
    pred = network(features)

    ind = features['ind_1']  # ind_1 => id of molecule for each atom
    nbatch = tf.reduce_max(ind)+1
    pred = tf.math.unsorted_segment_sum(pred, ind[:, 0], nbatch)

    if mode == tf.estimator.ModeKeys.TRAIN:
        n_trainable = np.sum([np.prod(v.shape)
                              for v in tf.compat.v1.trainable_variables()])
        print("Total number of trainable variables: {}".format(n_trainable))

        loss, metrics = _get_loss(features, pred, model_params)
        _make_train_summary(metrics)
        train_op = _get_train_op(loss,  model_params)
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        loss, metrics = _get_loss(features, pred, model_params)
        metrics = _make_eval_metrics(metrics)
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    if mode == tf.estimator.ModeKeys.PREDICT:
        pred = pred / model_params['e_scale']
        if model_params['e_dress']:
            pred += atomic_dress(features, model_params['e_dress'])
        pred *= model_params['e_unit']

        forces = -_get_dense_grad(pred, features['coord'])
        forces = tf.expand_dims(forces, 0)

        predictions = {
            'energy': pred,
            'forces': forces,
        }

        if 'cell' in features:
            stress = _get_dense_grad(pred, features['diff'])
            stress = tf.reduce_sum(
                tf.expand_dims(stress, 1) *
                tf.expand_dims(features['diff'], 2),
                axis=0, keepdims=True)
            stress /= tf.linalg.det(features['cell'])
            predictions['stress'] = stress
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions)


def _get_dense_grad(energy, coord):
    """get a gradient and convert to dense form"""
    import warnings
    index_warning = 'Converting sparse IndexedSlices'
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', index_warning)
        grad = tf.gradients(energy, coord)[0]
    if type(grad) == tf.IndexedSlices:
        grad = tf.scatter_nd(tf.expand_dims(grad.indices, 1), grad.values,
                             tf.cast(grad.dense_shape, tf.int32))
    return grad


@pi_named('LOSSES')
def _get_loss(features, pred, model_params):
    metrics = {}  # Not editting features here for safety, use a separate dict

    e_pred = pred
    e_data = features['e_data']
    if model_params['e_dress']:
        e_data -= atomic_dress(features, model_params['e_dress'])
    e_data *= model_params['e_scale']
    if model_params['max_energy']:
        # should get the mask here since max_energy refers to total energy
        e_mask = tf.abs(e_data) > model_params['max_energy']

    e_error = pred - e_data
    metrics['e_data'] = e_data
    metrics['e_pred'] = e_pred
    metrics['e_error'] = e_error

    if model_params['log_e_per_atom'] or model_params['use_e_per_atom']:
        ind_1 = features['ind_1']
        atom_count = tf.math.unsorted_segment_sum(
            tf.ones_like(ind_1, tf.float32), ind_1, tf.shape(e_data)[0])
        e_pred_per_atom = e_pred/atom_count
        e_data_per_atom = e_data/atom_count
        e_error_per_atom = e_error/atom_count
        metrics['e_data_per_atom'] = e_data_per_atom
        metrics['e_pred_per_atom'] = e_pred_per_atom
        metrics['e_error_per_atom'] = e_error_per_atom

    # e_error is ajusted from here
    if model_params['use_e_per_atom']:
        e_error = e_error_per_atom
        if model_params['use_e_per_sqrt']:
            e_error = e_error_per_atom*tf.sqrt(atom_count)
    if model_params['use_e_weight']:
        # Add this to metrics so that one can get a weighed RMSE
        metrics['e_weight'] = features['e_weight']
        e_error *= features['e_weight']
    if model_params['max_energy']:
        e_error = tf.where(e_mask, tf.zeros_like(e_error), e_error)
    # keep the per_sample loss so that it can be consumed by tf.compat.v1.metrics.mean
    e_loss = e_error**2 * model_params['e_loss_multiplier']
    metrics['e_loss'] = e_loss
    tot_loss = tf.reduce_mean(e_loss)

    if model_params['use_force']:
        f_pred = -_get_dense_grad(pred, features['coord'])
        f_data = features['f_data']*model_params['e_scale']
        f_error = f_pred - f_data
        metrics['f_data'] = f_data
        metrics['f_pred'] = f_pred
        metrics['f_error'] = f_error
        if model_params['use_f_weights']:
            f_error *= features['f_weights']
        if model_params['max_force']:
            f_error = tf.where(tf.abs(f_data) > model_params['max_force'],
                               tf.zeros_like(f_error), f_error)
        # keep the per_component loss here
        f_loss = f_error**2 * model_params['f_loss_multiplier']
        metrics['f_loss'] = f_loss
        tot_loss += tf.reduce_mean(f_loss)

    if model_params['use_stress']:
        s_pred = _get_dense_grad(pred, features['diff'])
        s_pred = tf.reduce_sum(
            tf.expand_dims(s_pred, 1) *
            tf.expand_dims(features['diff'], 2),
            axis=0, keepdims=True)
        s_pred /= tf.linalg.det(features['cell'])
        s_data = features['s_data']*model_params['e_scale']
        s_error = s_pred - s_data
        metrics['s_data'] = s_data
        metrics['s_pred'] = s_pred
        metrics['s_error'] = s_error
        s_loss = s_error**2 * model_params['s_loss_multiplier']
        metrics['s_loss'] = s_loss
        tot_loss += tf.reduce_mean(s_loss)

    if model_params['use_l2']:
        tvars = tf.compat.v1.trainable_variables()
        l2_loss = tf.add_n([
            tf.nn.l2_loss(v) for v in tvars if
            ('bias' not in v.name and 'E_OUT' not in v.name)])
        l2_loss = l2_loss * model_params['l2_loss_multiplier']
        metrics['l2_loss'] = l2_loss
        tot_loss += l2_loss

    metrics['tot_loss'] = tot_loss
    return tot_loss, metrics


@pi_named('METRICS')
def _make_eval_metrics(metrics):
    eval_metrics = {
        'METRICS/E_MAE': tf.compat.v1.metrics.mean_absolute_error(
            metrics['e_data'], metrics['e_pred']),
        'METRICS/E_RMSE': tf.compat.v1.metrics.root_mean_squared_error(
            metrics['e_data'], metrics['e_pred']),
        'METRICS/E_LOSS': tf.compat.v1.metrics.mean(metrics['e_loss']),
        'METRICS/TOT_LOSS': tf.compat.v1.metrics.mean(metrics['tot_loss'])
    }

    if 'e_data_per_atom' in metrics:
        eval_metrics['METRICS/E_PER_ATOM_MAE'] = tf.compat.v1.metrics.mean_absolute_error(
            metrics['e_data_per_atom'], metrics['e_pred_per_atom'])
        eval_metrics['METRICS/E_PER_ATOM_RMSE'] = tf.compat.v1.metrics.root_mean_squared_error(
            metrics['e_data_per_atom'], metrics['e_pred_per_atom'])

    if 'f_data' in metrics:
        eval_metrics['METRICS/F_MAE'] = tf.compat.v1.metrics.mean_absolute_error(
            metrics['f_data'], metrics['f_pred'])
        eval_metrics['METRICS/F_RMSE'] = tf.compat.v1.metrics.root_mean_squared_error(
            metrics['f_data'], metrics['f_pred'])
        eval_metrics['METRICS/F_LOSS'] = tf.compat.v1.metrics.mean(metrics['f_loss'])

    if 's_data' in metrics:
        eval_metrics['METRICS/S_MAE'] = tf.compat.v1.metrics.mean_absolute_error(
            metrics['s_data'], metrics['s_pred'])
        eval_metrics['METRICS/S_RMSE'] = tf.compat.v1.metrics.root_mean_squared_error(
            metrics['s_data'], metrics['s_pred'])
        eval_metrics['METRICS/S_LOSS'] = tf.compat.v1.metrics.mean(metrics['s_loss'])

    if 'l2_loss' in metrics:
        eval_metrics['METRICS/L2_LOSS'] = tf.compat.v1.metrics.mean(metrics['l2_loss'])
    return eval_metrics


@pi_named('METRICS')
def _make_train_summary(metrics):
    tf.compat.v1.summary.scalar('E_RMSE', tf.sqrt(tf.reduce_mean(metrics['e_error']**2)))
    tf.compat.v1.summary.scalar('E_MAE', tf.reduce_mean(tf.abs(metrics['e_error'])))
    tf.compat.v1.summary.scalar('E_LOSS', tf.reduce_mean(metrics['e_loss']))
    tf.compat.v1.summary.scalar('TOT_LOSS', metrics['tot_loss'])
    tf.compat.v1.summary.histogram('E_DATA', metrics['e_data'])
    tf.compat.v1.summary.histogram('E_PRED', metrics['e_pred'])
    tf.compat.v1.summary.histogram('E_ERROR', metrics['e_error'])

    if 'e_data_per_atom' in metrics:
        tf.compat.v1.summary.scalar(
            'E_PER_ATOM_MAE',
            tf.reduce_mean(tf.abs(metrics['e_error_per_atom'])))
        tf.compat.v1.summary.scalar(
            'E_PER_ATOM_RMSE',
            tf.sqrt(tf.reduce_mean(metrics['e_error_per_atom']**2)))
        tf.compat.v1.summary.histogram('E_PER_ATOM_DATA', metrics['e_data_per_atom'])
        tf.compat.v1.summary.histogram('E_PER_ATOM_PRED', metrics['e_pred_per_atom'])
        tf.compat.v1.summary.histogram('E_PER_ATOM_ERROR', metrics['e_error_per_atom'])

    if 'f_data' in metrics:
        tf.compat.v1.summary.scalar('F_MAE', tf.reduce_mean(tf.abs(metrics['f_error'])))
        tf.compat.v1.summary.scalar('F_RMSE', tf.sqrt(
            tf.reduce_mean(metrics['f_error']**2)))
        tf.compat.v1.summary.scalar('F_LOSS', tf.reduce_mean(metrics['f_loss']))
        tf.compat.v1.summary.histogram('F_DATA', metrics['f_data'])
        tf.compat.v1.summary.histogram('F_PRED', metrics['f_pred'])
        tf.compat.v1.summary.histogram('F_ERROR', metrics['f_error'])

    if 's_data' in metrics:
        tf.compat.v1.summary.scalar('S_MAE', tf.reduce_mean(tf.abs(metrics['s_error'])))
        tf.compat.v1.summary.scalar('S_RMSE', tf.sqrt(
            tf.reduce_mean(metrics['s_error']**2)))
        tf.compat.v1.summary.scalar('S_LOSS', tf.reduce_mean(metrics['s_loss']))
        tf.compat.v1.summary.histogram('S_DATA', metrics['s_data'])
        tf.compat.v1.summary.histogram('S_PRED', metrics['s_pred'])
        tf.compat.v1.summary.histogram('S_ERROR', metrics['s_error'])

    if 'l2_loss' in metrics:
        tf.compat.v1.summary.scalar('L2_LOSS', metrics['l2_loss'])


@pi_named('TRAIN_OP')
def _get_train_op(loss, model_params):
    # Get the optimizer
    global_step = tf.compat.v1.train.get_global_step()
    learning_rate = model_params['learning_rate']
    if model_params['use_decay']:
        learning_rate = tf.compat.v1.train.exponential_decay(
            learning_rate, global_step,
            model_params['decay_step'], model_params['decay_rate'],
            staircase=True)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
    # Get the gradients
    tvars = tf.compat.v1.trainable_variables()
    grads = tf.gradients(loss, tvars)
    if model_params['use_norm_clip']:
        grads, _ = tf.clip_by_global_norm(grads, model_params['norm_clip'])
    return optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

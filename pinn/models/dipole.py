# -*- coding: utf-8 -*-
"""Models interfaces neural networks with the estimator api
Models transfer a network to a model function to use with the tf estimator api.
A model defines the goal/loss of the model, as well as training paramters.
"""
import tensorflow as tf
import numpy as np

from pinn import get_network
from pinn.utils import pi_named
from pinn.models.base import export_model

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
    'log_d_per_atom': False,  # log d_per_atom and its distribution
                             # ^- this is forcely done if use_d_per_atom
    'use_d_weight': False,   # scales the loss according to d_weight
    # Loss function multipliers
    'd_loss_multiplier': 1.0,
}

@export_model
def dipole_model(features, labels, mode, params):
    """Model function for neural network dipoles"""
    network = get_network(params['network'])
    model_params = default_params
    model_params.update(params['model']['params'])

    features = network.preprocess(features)
    connect_dist_grad(features)
    pred = network(features)
    pred = tf.expand_dims(pred, axis=1)
    ind = features['ind_1']  # ind_1 => id of molecule for each atom
    nbatch = tf.reduce_max(ind)+1
    charge = tf.math.unsorted_segment_sum(pred, ind[:, 0], nbatch)
    dipole = pred * features['coord']
    dipole = tf.math.unsorted_segment_sum(dipole, ind[:, 0], nbatch)
    dipole = tf.sqrt(tf.reduce_sum(dipole**2, axis=1)+1e-6)

    if mode == tf.estimator.ModeKeys.TRAIN:
        metrics = make_metrics(features, dipole, charge, model_params, mode)
        train_op = get_train_op(params['optimizer'],
                                metrics.LOSS, metrics.ERROR, network)
        return tf.estimator.EstimatorSpec(mode, loss=metrics.LOSS, train_op=train_op)
    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = make_metrics(features, dipole, charge, model_params, mode)
        return tf.estimator.EstimatorSpec(mode, loss=metrics.LOSS,
                                          eval_metric_ops=metrics.METRICS)
    else:
        pred = pred / model_params['d_scale']
        pred *= model_params['d_unit']

        predictions = {
            'dipole': dipole,
            'charges': tf.expand_dims(pred, 0)
        }
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions)


@pi_named("METRICS")
def make_metrics(features, d_pred, q_pred, params, mode):
    from pinn.models.base import MetricsCollector, get_train_op

    d_data = features['d_data']
    q_data = tf.zeros_like(q_pred)
    d_data *= model_params['d_scale']
    d_mask = tf.abs(d_data) > params['max_dipole'] if params['max_dipole'] else None
    d_weight = params['d_loss_multiplier']
    d_weight *= features['d_weight'] if params['use_d_weight'] else 1

    metrics.add_error('Q', q_data, q_pred)
    metrics.add_error('D', d_data, d_pred, mask=d_mask, weight=d_weight,
                      use_error=(not params['use_d_per_atom']))

    if params['use_d_per_atom'] or params['log_d_per_atom']:
        n_atoms = count_atoms(features['ind_1'])
        metrics.add_error('D_PER_ATOM', d_data/n_atoms, d_pred/n_atoms, mask=d_mask,
                          weight=d_weight, use_error=params['use_d_per_atom'],
                          log_error=params['log_d_per_atom'])

    return metrics

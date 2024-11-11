# -*- coding: utf-8 -*-
r"""This file implements the atomic dipole (AD) dipole model.

Atomic vectorial property predictions from the network are interpreted as atomic dipoles.
The dipole moment is expressed as: 

$$
\begin{aligned}
\mu = \sum_{i}{}^{3}\mathbb{P}_{i}
\end{aligned}
$$

This model fits the total dipole of the inputs and predicts the total dipole.
For model details see ref. 
Li, J., Knijff, L., Zhang, Z., Andersson, L., & Zhang, C. (2024)
PiNN: equivariant neural network suite for modelling electrochemical systems
"""
import numpy as np
import tensorflow as tf
from pinn import get_network
from pinn.utils import pi_named, make_indices
from pinn.models.base import export_model, get_train_op, MetricsCollector

default_params = {
    ### Scaling and units
    # The loss function will be MSE((pred - label) * scale)
    # For vector/tensor predictions
    # the error will be pre-component instead of per-atom
    # d_unit is the unit of dipole to report w.r.t the input labels
    'd_scale': 1.0,  # dipole scale for prediction
    'd_unit': 1.0,  # output unit of dipole during prediction
    # Toggle whether to use scalar or vector dipole predictions
    'vector_dipole': False,
    # Loss function options
    'max_dipole': False,     # if set to float, omit dipoles larger than it
    'use_d_per_atom': False,  # use d_per_atom to calculate d_loss
    'log_d_per_atom': False,  # log d_per_atom and its distribution
                             # ^- this is forcely done if use_d_per_atom
    'use_d_weight': False,   # scales the loss according to d_weight
    # L2 loss
    'use_l2': False,
    # Loss function multipliers
    'd_loss_multiplier': 1.0,
    'q_loss_multiplier': 1.0
}

@export_model
def AD_dipole_model(features, labels, mode, params):
    r"""The atomic dipole (AD) dipole model constructs the dipole moment from 
    atomic dipole predictions:

    $$
    \begin{aligned}
    \mu = \sum_{i}{}^{3}\mathbb{P}_{i}
    \end{aligned}
    $$
    """
    network = get_network(params['network'])
    model_params = default_params.copy()
    model_params.update(params['model']['params'])

    features = network.preprocess(features)
    p1, output_p3 = network(features)
    p3 = output_p3['p3']
    p3 = tf.squeeze(p3, axis=-1)
    
    ind1 = features['ind_1']  # ind_1 => id of molecule for each atom
    ind2 = features['ind_2']

    natoms = tf.reduce_max(tf.shape(ind1))
    nbatch = tf.reduce_max(ind1)+1 

    atomic_d = tf.math.unsorted_segment_sum(p3, ind1[:, 0], nbatch)

    dipole = atomic_d
  
    if model_params['vector_dipole'] == False:
        dipole = tf.sqrt(tf.reduce_sum(dipole**2, axis=1)+1e-6)
  
    if mode == tf.estimator.ModeKeys.TRAIN:
        metrics = make_metrics(features, dipole, model_params, mode)
        tvars = network.trainable_variables
        train_op = get_train_op(params['optimizer'], metrics, tvars)
        return tf.estimator.EstimatorSpec(mode, loss=tf.reduce_sum(metrics.LOSS),
                                          train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = make_metrics(features, dipole, model_params, mode)
        return tf.estimator.EstimatorSpec(mode, loss=tf.reduce_sum(metrics.LOSS),
                                          eval_metric_ops=metrics.METRICS)
    else:
        dipole = dipole / model_params['d_scale']
        dipole *= model_params['d_unit']

        predictions = {
            #'dipole': dipole,
            'atomic_d': tf.expand_dims(p3, 0)
        }
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions)


@pi_named("METRICS")
def make_metrics(features, d_pred, params, mode):
    metrics = MetricsCollector(mode)

    d_data = features['d_data']
    d_data *= params['d_scale']
    d_mask = tf.abs(d_data) > params['max_dipole'] if params['max_dipole'] else None
    d_weight = params['d_loss_multiplier']
    d_weight *= features['d_weight'] if params['use_d_weight'] else 1

    metrics.add_error('D', d_data, d_pred, mask=d_mask, weight=d_weight,
                      use_error=(not params['use_d_per_atom']))

    if params['use_d_per_atom'] or params['log_d_per_atom']:
        n_atoms = count_atoms(features['ind_1'], dtype=d_data.dtype)
        metrics.add_error('D_PER_ATOM', d_data/n_atoms, d_pred/n_atoms, mask=d_mask,
                          weight=d_weight, use_error=params['use_d_per_atom'],
                          log_error=params['log_d_per_atom'])

    if params['use_l2']:
        tvars = tf.compat.v1.trainable_variables()
        l2_loss = tf.add_n([
            tf.nn.l2_loss(v) for v in tvars if
            ('bias' not in v.name and 'noact' not in v.name)])
        l2_loss = l2_loss * params['l2_loss_multiplier']
        metrics.METRICS['METRICS/L2_LOSS'] = l2_loss
        metrics.LOSS.append(l2_loss)

    return metrics

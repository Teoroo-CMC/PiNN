# -*- coding: utf-8 -*-
"""This file implements the potential model

Atomic predictions from the network are interpreted as atomic energies. This
model is also capable of fitting and predicting forces and stresses.
"""
import numpy as np
import tensorflow as tf
from pinn import get_network
from pinn.utils import pi_named, atomic_dress, connect_dist_grad
from pinn.models.base import export_model, get_train_op, MetricsCollector

default_params = {
    ### Scaling and units # The loss function will be MSE((pred - label) * scale)
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
    'max_energy': False,        # if set to float, omit energies larger than it
    'use_e_per_atom': False,    # use e_per_atom to calculate e_loss
    'log_e_per_atom': False,    # log e_per_atom and its distribution
                                # ^- this is forcely done if use_e_per_atom
    'use_e_weight': False,      # scales the loss according to e_weight
    ## Force
    'use_force': False,         # include force in loss function
    'max_force_comp': False,    # if set to float, omit forces components larger than it
    'no_force_comp': False,     # if set to int, use as maximum number of force components for a update
    'use_f_weights': False,      # scales the loss according to f_weights
    ## Stress
    'use_stress': False,        # include stress in Loss function
    ## L2
    'use_l2': False,
    ## Loss function multipliers
    'e_loss_multiplier': 1.0,
    'f_loss_multiplier': 1.0,
    's_loss_multiplier': 1.0,
    'l2_loss_multiplier': 1.0,
    'separate_errors': False,   # workaround at this point
}

@export_model
def potential_model(features, labels, mode, params):
    """Model function for neural network potentials"""
    network = get_network(params['network'])
    model_params = default_params.copy()
    model_params.update(params['model']['params'])

    features = network.preprocess(features)
    connect_dist_grad(features)
    pred = network(features)

    ind = features['ind_1']
    nbatch = tf.reduce_max(ind)+1
    pred = tf.math.unsorted_segment_sum(pred, ind[:, 0], nbatch)

    if mode == tf.estimator.ModeKeys.TRAIN:
        metrics = make_metrics(features, pred, model_params, mode)
        tvars = network.trainable_variables
        train_op = get_train_op(params['optimizer'], metrics, tvars,
                                separate_errors=model_params['separate_errors'])
        return tf.estimator.EstimatorSpec(mode, loss=tf.reduce_sum(metrics.LOSS),
                                          train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = make_metrics(features, pred, model_params, mode)
        return tf.estimator.EstimatorSpec(mode, loss=tf.reduce_sum(metrics.LOSS),
                                          eval_metric_ops=metrics.METRICS)

    if mode == tf.estimator.ModeKeys.PREDICT:
        pred = pred / model_params['e_scale']
        if model_params['e_dress']:
            pred += atomic_dress(features, model_params['e_dress'], dtype=pred.dtype)
        pred *= model_params['e_unit']
        forces = -_get_dense_grad(pred, features['coord'])
        forces = tf.expand_dims(forces, 0)
        predictions = {'energy': pred, 'forces': forces}
        if 'cell' in features:
            stress = _get_stress(pred, features)
            predictions['stress'] = stress
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

@pi_named("METRICS")
def make_metrics(features, pred, params, mode):
    from pinn.utils import count_atoms

    metrics = MetricsCollector(mode)

    e_pred = pred
    e_data = features['e_data']
    if params['e_dress']:
        e_data -= atomic_dress(features, params['e_dress'], dtype=pred.dtype)
    e_data *= params['e_scale']

    # should get the mask here since max_energy refers to total energy
    e_mask = tf.abs(e_data) > params['max_energy'] if params['max_energy'] else None
    e_weight = params['e_loss_multiplier']
    e_weight *= features['e_weight'] if params['use_e_weight'] else 1
    metrics.add_error('E', e_data, e_pred, mask=e_mask, weight=e_weight,
                      use_error=(not params['use_e_per_atom']))

    if params['use_e_per_atom'] or params['log_e_per_atom']:
        n_atoms = count_atoms(features['ind_1'], dtype=e_data.dtype)
        metrics.add_error('E_PER_ATOM', e_data/n_atoms, e_pred/n_atoms, mask=e_mask,
                          weight=e_weight, use_error=params['use_e_per_atom'],
                          log_error=params['log_e_per_atom'])

    if params['use_force']:
        f_pred = -_get_dense_grad(pred, features['coord'])
        f_data = features['f_data']*params['e_scale']
        f_mask = tf.fill(tf.shape(f_pred), True)

        if params['max_force_comp']:
            f_mask = tf.abs(f_data)<params['max_force']
            f_pred = tf.boolean_mask(f_pred, use_ind)
            f_data = tf.boolean_mask(f_data, use_ind)

        if params['no_force_comp']:
            use_ind = tf.cast(tf.random.shuffle(tf.where(f_mask))[:params['no_force_comp']], tf.int32)
            f_mask = tf.scatter_nd(use_ind, tf.fill(tf.shape(use_ind)[:1],True), tf.shape(f_mask))

        f_weight = params['f_loss_multiplier']
        f_weight *= features['f_weights'] if params['use_f_weights'] else 1
        metrics.add_error('F', f_data, f_pred, mask=f_mask, weight=f_weight)

    if params['use_stress']:
        s_pred = _get_stress(pred, features)
        s_data = features['s_data']*params['e_scale']
        metrics.add_error('S', s_data, s_pred, weight=params['s_loss_multiplier'])

    if params['use_l2']:
        tvars = tf.compat.v1.trainable_variables()
        l2_vars = tf.concat([
            tf.reshape(v, [-1]) for v in tvars if
            ('bias' not in v.name and 'noact' not in v.name)], axis=0)
        metrics.add_error('L2', l2_vars, 0, weight=params['l2_loss_multiplier'],
                          log_error=False, log_hist=False)

    return metrics


def _get_stress(pred, tensors):
    f_ij = _get_dense_grad(pred, tensors['diff'])
    s_pred = tf.reduce_sum(
        tf.expand_dims(f_ij, 1) *
        tf.expand_dims(tensors['diff'], 2),
        axis=0, keepdims=True)
    s_pred /= tf.linalg.det(tensors['cell'])
    return s_pred


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

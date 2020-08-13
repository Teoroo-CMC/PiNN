# -*- coding: utf-8 -*-
"""Models interfaces neural networks with the estimator api

Models transfer a network to a model function to use with the tf estimator api.
A model defines the goal/loss of the model, as well as training paramters.
"""
import tensorflow as tf
import numpy as np

from pinn import get_network
from pinn.layers import atomic_dress
from pinn.utils import pi_named, connect_dist_grad
from pinn.models.base import export_model, get_train_op, MetricsCollector

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
    'max_energy': False,        # if set to float, omit energies larger than it
    'use_e_per_atom': False,    # use e_per_atom to calculate e_loss
    'use_e_per_sqrt': False,
    'log_e_per_atom': False,    # log e_per_atom and its distribution
                                # ^- this is forcely done if use_e_per_atom
    'use_e_weight': False,      # scales the loss according to e_weight
    ## Force
    'use_force': False,         # include force in loss function
    'use_single_force': False,  # use single force during weight updates
    'max_force': False,         # if set to float, omit forces larger than it
    'use_f_weight': False,      # scales the loss according to f_weights
    ## Stress
    'use_stress': False,        # include stress in Loss function
    # Loss function multipliers
    'e_loss_multiplier': 1.0,
    'f_loss_multiplier': 1.0,
    's_loss_multiplier': 1.0,
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
        train_op = get_train_op(params['optimizer'],
                                metrics.LOSS, metrics.ERROR, network)
        return tf.estimator.EstimatorSpec(mode, loss=metrics.LOSS, train_op=train_op)
    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = make_metrics(features, pred, model_params, mode)
        return tf.estimator.EstimatorSpec(mode, loss=metrics.LOSS,
                                          eval_metric_ops=metrics.METRICS)
    if mode == tf.estimator.ModeKeys.PREDICT:
        pred = pred / model_params['e_scale']
        if model_params['e_dress']:
            pred += atomic_dress(features, model_params['e_dress'])
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
        e_data -= atomic_dress(features, params['e_dress'])
    e_data *= params['e_scale']

    # should get the mask here since max_energy refers to total energy
    e_mask = tf.abs(e_data) > params['max_energy'] if params['max_energy'] else None
    e_weight = params['e_loss_multiplier']
    e_weight *= features['e_weight'] if params['use_e_weight'] else 1
    metrics.add_error('E', e_data, e_pred, mask=e_mask, weight=e_weight,
                      use_error=(not params['use_e_per_atom']))

    if params['use_e_per_atom'] or params['log_e_per_atom']:
        n_atoms = count_atoms(features['ind_1'])
        metrics.add_error('E_PER_ATOM', e_data/n_atoms, e_pred/n_atoms, mask=e_mask,
                          weight=e_weight, use_error=params['use_e_per_atom'],
                          log_error=params['log_e_per_atom'])

    if params['use_force']:
        f_pred = -_get_dense_grad(pred, features['coord'])
        f_data = features['f_data']*params['e_scale']
        if params['use_single_force']:
            all_ind = tf.shape(features['ind_1'])[0]
            use_ind = tf.random.uniform([], maxval= all_ind, dtype=tf.int32)
            f_pred = f_pred[use_ind, :]
            f_data = f_data[use_ind, :]
        f_mask = tf.abs(f_data) > params['max_force'] if params['max_force'] else None
        f_weight = params['f_loss_multiplier']
        f_weight *= features['f_weight'] if params['use_f_weight'] else 1
        metrics.add_error('F', f_data, f_pred, mask=f_mask, weight=f_weight,
                          use_error=params['use_force'], log_error=params['use_force'])

    if params['use_stress']:
        s_pred = _get_stress(pred, features)
        s_data = features['s_data']*params['e_scale']
        metrics.add_error('S', s_data, s_pred, weight=params['s_loss_multiplier'],
                          use_error=params['use_stress'], log_error=params['use_stress'])
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

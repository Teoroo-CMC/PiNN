# -*- coding: utf-8 -*-
"""Models interfaces neural networks with the estimator api

Models transfer a network to a model function to use with the tf estimator api.
A model defines the goal/loss of the model, as well as training paramters.
"""

import tensorflow as tf
import pinn.filters as f
import pinn.layers as l
import pinn.networks
from pinn.utils import pi_named


def potential_model(params, config=None):
    """Shortcut for generating potential model from paramters

    Args:
        params: a dictionary specifing the model
        config: tensorflow config for the estimator
    """
    model = tf.estimator.Estimator(
        model_fn=_potential_model_fn, params=params,
        model_dir=params['model_dir'], config=config)
    return model


def _potential_model_fn(features, labels, mode, params):
    """Model function for neural network potentials"""
    if isinstance(params['network'], str):
        network_fn = getattr(pinn.networks, params['network'])
    else:
        network_fn = params['network']
    net_param = params['netparam']
    train_param = _get_train_param(params['train'])
    if 'train_force' not in train_param:
        train_param['train_force'] = False
        
    pred = network_fn(features, **net_param)
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_global_step()
        
        loss = _get_loss(features, pred, train_param)
        _get_train_summary(features, pred, train_param)
        train_op = _get_train_op(loss, global_step, train_param)
        
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        loss = _get_loss(features, pred, train_param)
        metrics = _get_metrics(features, pred, train_param)
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    if mode == tf.estimator.ModeKeys.PREDICT:
        forces = _get_forces(pred, features['coord'])
        forces = tf.expand_dims(forces, 0)
        stress = _get_forces(pred, features['diff'])
        stress = tf.reduce_sum(
            tf.expand_dims(stress,1)*
            tf.expand_dims(features['diff'],2),
            axis=0, keepdims=True)
        predictions = {
            'energy': pred,
            'forces': -forces,
            'stress': stress
        }
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions)


def _reset_dist_grad(tensors):
    tensors['diff'] = _connect_diff_grad(tensors['coord'], tensors['diff'],
                                         tensors['ind'][2])
    tensors['dist'] = _connect_dist_grad(tensors['diff'], tensors['dist'])

@tf.custom_gradient
def _connect_diff_grad(coord, diff, ind):
    """Returns a new diff with its gradients connected to coord"""
    def _grad(ddiff, coord, diff, ind):
        natoms = tf.shape(coord)[0]
        dcoord = tf.unsorted_segment_sum(ddiff, ind[:,1], natoms)
        dcoord -= tf.unsorted_segment_sum(ddiff, ind[:,0], natoms)
        return dcoord, None, None, None
    return tf.identity(diff), lambda ddist: _grad(ddiff, coord, diff, ind)

@tf.custom_gradient
def _connect_dist_grad(diff, dist):
    """Returns a new dist with its gradients connected to diff"""
    def _grad(ddist, diff, dist):
        return tf.expand_dims(ddist/dist, 1)*diff, None
    return tf.identity(dist), lambda ddist: _grad(ddist, diff, dist)    
    
def _get_forces(energy, coord):
    import warnings
    index_warning = 'Converting sparse IndexedSlices'
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', index_warning)
        force = tf.gradients(energy, coord)[0]
    if type(force) == tf.IndexedSlices:
        force = tf.scatter_nd(tf.expand_dims(force.indices, 1),
                              force.values, tf.cast(force.dense_shape, tf.int32))
    return force

@pi_named('LOSSES')
def _get_loss(features, pred, train_param):
    if 'e_dress'in features:
        features['e_data'] = features['e_data'] - features['e_dress']
    features['e_data'] = features['e_data'] * train_param['en_scale']
    loss = tf.losses.mean_squared_error(features['e_data'], pred)
    if train_param['train_force']:
        features['f_data'] = features['f_data'] * train_param['en_scale']
        features['forces'] = -_get_forces(pred, features['coord'])
        frc_loss = tf.losses.mean_squared_error(
            features['f_data'], features['forces'])
        loss = loss + train_param['force_ratio'] * frc_loss
    tvars = tf.trainable_variables()
    if train_param['regularize_l2']:
        loss += train_param['regularize_l2'] * tf.add_n([
            tf.nn.l2_loss(v) for v in tvars if
            ('bias' not in v.name and 'energy' not in v.name)]) 
    return loss

@pi_named('METRICS')
def _get_metrics(features, pred, train_param):
    metrics = {
        'METRICS/ENG_MAE': tf.metrics.mean_absolute_error(

            features['e_data'], pred),
        'METRICS/ENG_RMSE': tf.metrics.root_mean_squared_error(
            features['e_data'], pred)}
    if train_param['train_force']:
        metrics['METRICS/FRC_MAE'] = tf.metrics.mean_absolute_error(
            features['f_data'], features['forces'])
        metrics['METRICS/FRC_RMSE'] = tf.metrics.root_mean_squared_error(
            features['f_data'], features['forces'])
    return metrics

@pi_named('TRAIN_OP')
def _get_train_op(loss, global_step, train_param):
    learning_rate = train_param['learning_rate']
    if train_param['decay']:
        learning_rate = tf.train.exponential_decay(
            learning_rate, global_step,
            train_param['decay_step'], train_param['decay_rate'], 
            staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)
    if train_param['norm_clip']:
        grads, _ = tf.clip_by_global_norm(grads, train_param['norm_clip'])
    train_op = optimizer.apply_gradients(
        zip(grads, tvars), global_step=global_step)
    return train_op


@pi_named('METRICS')
def _get_train_summary(features, pred, train_param):
    tf.summary.scalar(
        'ENG_RMSE', tf.sqrt(tf.losses.mean_squared_error(
            features['e_data'], pred)))
    tf.summary.scalar(
        'ENG_MAE', tf.reduce_mean(tf.abs(
            features['e_data'] - pred)))
    tf.summary.histogram('E_DATA', features['e_data'])
    tf.summary.histogram('E_PRED', pred)
    tf.summary.histogram('E_ERROR', features['e_data'] - pred)

    if train_param['train_force']:
        tf.summary.scalar(
            'FRC_RMSE', tf.sqrt(tf.losses.mean_squared_error(
                features['f_data'], features['forces'])))
        tf.summary.scalar(
            'FRC_MAE', tf.reduce_mean(tf.abs(
                features['f_data'] - features['forces'])))
        tf.summary.histogram('F_DATA', features['f_data'])
        tf.summary.histogram('F_PRED', features['forces'])
        tf.summary.histogram(
            'F_ERROR', features['f_data']-features['forces'])

        
def _get_train_param(train_param):
    default_param = {
        'en_scale': 1,
        'learning_rate': 3e-4,
        'norm_clip': 0.01,
        'decay': True,
        'regularize_l2': 0.001,
        'decay_step':100000,
        'decay_rate':0.96}
    for k, v in default_param.items():
        if k not in train_param:
            train_param[k]=v
    return train_param

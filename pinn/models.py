# -*- coding: utf-8 -*-
"""Models interfaces neural networks with the estimator api

Models transfer a network to a model function to use with the tf estimator api.
A model defines the goal/loss of the model, as well as training paramters.
"""

import tensorflow as tf
import pinn.filters as f
import pinn.layers as l
import pinn.networks

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
    return loss

def _get_metrics(features, pred, train_param):
    metrics = {
        'ENG_MAE': tf.metrics.mean_absolute_error(
            features['e_data'], pred),
        'ENG_RMSE': tf.metrics.root_mean_squared_error(
            features['e_data'], pred)}
    if train_param['train_force']:
        metrics['FRC_MAE'] = tf.metrics.mean_absolute_error(
            features['f_data'], features['forces'])
        metrics['FRC_RMSE'] = tf.metrics.root_mean_squared_error(
            features['f_data'], features['forces'])
    return metrics


def _get_train_op(loss, global_step, train_param):
    learning_rate = train_param['learning_rate']
    regularization = train_param['regularization']
    learning_rate = tf.train.exponential_decay(
        learning_rate, global_step, 100000, 0.96, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)
    if regularization=='clip':
        grads, _ = tf.clip_by_global_norm(grads, 0.01)
    train_op = optimizer.apply_gradients(
        zip(grads, tvars), global_step=global_step)
    return train_op


def _potential_model_fn(features, labels, mode, params):
    """Model function for neural network potentials"""
    if isinstance(params['network']['func'], str):
        network_fn = getattr(pinn.networks, params['network']['func'])
    else:
        network_fn = params['network']['func']
    net_param = params['network']['params']
    train_param = params['train']
    if 'train_force' not in train_param:
        train_param['train_force'] = False
    pred = network_fn(features, **net_param)

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_global_step()
        loss = _get_loss(features, pred, train_param)

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

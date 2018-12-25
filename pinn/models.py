# -*- coding: utf-8 -*-
"""Models interfaces neural networks with the estimator api

Models transfer a network to a model function to use with the tf estimator api.
A model defines the goal/loss of the model, as well as training paramters.
"""

import tensorflow as tf
import pinn.filters as f
import pinn.layers as l
import pinn

def _get_forces(energy, coord):
    import warnings
    index_warning = 'Converting sparse IndexedSlices'
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', index_warning)
        force = tf.gradients(energy, coord)[0]
    if type(force) == tf.IndexedSlices:
        force = tf.scatter_nd(tf.expand_dims(force.indices, 1),
                              force.values, tf.cast(force.dense_shape, tf.int32))
    print(force.shape)
    return force
    
def _get_loss(features, pred, scale=None):
    if 'e_dress' in features:
        features['e_data'] = features['e_data'] - features['e_dress']
    if scale is not None:
        features['e_data'] = features['e_data'] * scale
    loss = tf.losses.mean_squared_error(features['e_data'], pred)
    return loss

def _get_train_op(loss, global_step, learning_rate=1e-4,
                  regularization=None):
    learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                                              100000, 0.96, staircase=True)
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
    if mode == tf.estimator.ModeKeys.PREDICT:
        net_param['pre_level'] = 0

    pred = network_fn(features, **net_param)

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_global_step()
        loss = _get_loss(features, pred, train_param)
        train_op = _get_train_op(loss, global_step, train_param)
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        loss = _get_loss(features, pred, train_param)
        metrics = {
            'MAE': tf.metrics.mean_absolute_error(
                features['e_data'], pred),
            'RMSE': tf.metrics.root_mean_squared_error(
                features['e_data'], pred)}
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # if 'atomic_dress' in tensors:
        #     pred = pred + features['atomic_dress']
        print(pred)
        if 'e_scale' in model_param['loss']:
            pred = pred * model_param['loss']['e_scale']
        forces = _get_forces(pred, features['coord'])
        forces = tf.expand_dims(forces, 0)
        stress = _get_forces(pred, features['diff'])
        stress = tf.reduce_sum(
            tf.expand_dims(stress,1)*
            tf.expand_dims(features['diff'],2),
            axis=0, keepdims=True)
        predictions = {
            'energy': pred,
            'forces': - forces,
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

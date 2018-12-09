# -*- coding: utf-8 -*-
"""Models interfaces neural networks with the estimator api

Models transfer a network to a model function to use with the tf estimator api.
A model defines the goal/loss of the model, as well as training paramters.
"""

import tensorflow as tf
import pinn.filters as f
import pinn.layers as l


def potential_model_fn(features, labels, mode, params):
    """Model function for neural network potentials

    Args:
        params: a dictionary specifing the model
        config: tensorflow config for the estimator
    """
    network_fn = getattr(pinn.networks, params['network']['func'])
    net_param = params['network']['params']
    model_param = params['model']
    pred = network_fn(features, **net_param)

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_global_step()
        loss = get_loss(**model_param['loss'])(features, pred)
        train_op = get_train_op(**model_param['train'])
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        loss = get_loss(**model_param['loss'])(features, pred)
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    if mode == tf.estimator.ModeKeys.PREDICT:
        if 'atomic_dress' in tensors:
            pred = pred + features['atomic_dress']
        if 'e_scale' in model_param['loss']:
            pred = pred * model_param['loss']['e_scale']
        predictions = {
            'energy': pred,
            'forces': l.get_forces(pred, features['coord'])
            'stress': l.get_stress(pred, features['diff'])}
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions)

def potential_model(params, config=None):
    """Shortcut for generating potential model from paramters
    """
    model = tf.estimator.Estimator(
        model_fn=potential_model_fn, params=params,
        model_dir=params['model_dir'], config=config)
    return model

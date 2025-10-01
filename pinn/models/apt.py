import numpy as np
import tensorflow as tf
from pinn import get_network
from pinn.utils import pi_named
from pinn.models.base import export_model, get_train_op, MetricsCollector

@pi_named("METRICS")
def make_metrics(features, predictions, params, mode):
    from pinn.utils import count_atoms
    metrics = MetricsCollector(mode)
    pred = predictions
    data = features['apt']
    error = pred - data
    metrics.add_error('apt_error', pred, data, log_error=True)
    return metrics

default_params = {
    ### Scaling and units
}

@export_model
def apt_model(features, labels, mode, params):
    network = get_network(params['network'])
    model_params = default_params.copy()
    model_params.update(params['model']['params'])

    features = network.preprocess(features)

    predictions = apt_function(features,network)
    pred = predictions['apt']
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        metrics = make_metrics(features, pred, model_params, mode)
        tvars = network.trainable_variables
        train_op = get_train_op(params['optimizer'], metrics, tvars)
        return tf.estimator.EstimatorSpec(mode, loss=tf.reduce_sum(metrics.LOSS),
                                          train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = make_metrics(features, pred, model_params, mode)
        return tf.estimator.EstimatorSpec(mode, loss=tf.reduce_sum(metrics.LOSS),
                                          eval_metric_ops=metrics.METRICS)

    if mode == tf.estimator.ModeKeys.PREDICT:
        apt_outer = predictions['apt_outer']
        apt_iso = predictions['apt_iso']
        predictions = {'apt': pred, 'apt_outer': apt_outer, 'apt_iso': apt_iso}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

def apt_function(features,network):
    # This will crash if nbatch>1
    p1, output_dict = network(features)
    p3 = output_dict['p3']
    p3 = tf.squeeze(p3, axis=-1)
    i3 = output_dict['i3']
    i3 = tf.squeeze(i3, axis=-1)
    
    ind1 = features['ind_1']
    ind2 = features['ind_2']
    natoms = tf.reduce_max(tf.shape(ind1))
    nbatch = tf.reduce_max(ind1)+1

    apt_outer_pair = tf.einsum('ix,iy->ixy',features['diff'],i3)
    apt_outer = tf.math.unsorted_segment_sum(apt_outer_pair, ind2[:, 0], natoms)
    i = tf.eye(3,batch_shape=[natoms])
    apt_iso = tf.linalg.set_diag(i,p3)
    apt_iso = tf.reshape(apt_iso,[nbatch,natoms,3,3])
    apt_outer = tf.reshape(apt_outer,[nbatch,natoms,3,3])
    apt = apt_iso + apt_outer
    qcorr = tf.reduce_sum(apt,axis=1,keepdims=True)/tf.cast(natoms,tf.float32)
    apt -= qcorr 

    return {'apt': apt, 'apt_outer': apt_outer, 'apt_iso': apt_iso}

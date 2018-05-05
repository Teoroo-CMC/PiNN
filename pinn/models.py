"""
   Models are defined atomic neural networks
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   Models are built upon the tensorflow Estimator API
   Implemented models:
      PiNN: Atomic Neural Network potential based on pairwise interactions
"""
import tensorflow as tf
import pinn.filters as f
import pinn.layers as l


def potential_model_fn(features, labels, mode, params):
    """
    """
    for layer in params['filters']:
        layer.parse(features, dtype=params['dtype'])

    for layer in params['layers']:
        layer.parse(features, dtype=params['dtype'])

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_global_step()
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)

        loss = tf.losses.mean_squared_error(features['e_data'],
                                            features['energy'])
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=optimizer.minimize(loss, global_step=global_step))

    if mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.losses.mean_squared_error(features['e_data'],
                                            features['energy'])

        print(features['e_data'].shape)
        print(features['energy'].shape)
        metrics = {
            'MAE': tf.metrics.mean_absolute_error(
                features['e_data'], features['energy']),
            'RMSE': tf.metrics.root_mean_squared_error(
                features['e_data'], features['energy'])
        }

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops=metrics)



def PiNN(depth=6, p_nodes=32, i_nodes=8, act='tanh', rc=4.0):
    """"""
    filters = [
        f.atomic_mask(),
        f.atomic_dress({0: 0.0}),
        f.distance_mat(),
        f.pi_kernel(),
        f.pi_atomic([1, 6, 7, 8, 9])
    ]

    layers = [
        l.fc_layer('fc_01', order=0, n_nodes=[p_nodes], act=act),
        l.pi_layer('pi_01', order=1, n_nodes=[i_nodes], act=act),
        l.ip_layer('ip_01', order=1, pool_type='sum'),
        l.en_layer('en_01', order=1, n_nodes=[p_nodes], act=act)]

    params = {
        'filters': filters,
        'layers': layers,
        'prefetch_level': 0,
        'batch_size': 100,
        'dtype': tf.float32
    }

    estimator = tf.estimator.Estimator(
        model_fn=potential_model_fn, params=params)
    return estimator

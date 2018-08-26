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
        if mode == tf.estimator.ModeKeys.TRAIN and layer.name.startswith('pi'):
            tf.summary.image(layer.name, features['nodes'][1].get_dense()[:,:,:,0:3])

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_global_step()
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)

        loss = tf.losses.mean_squared_error(features['e_data'],
                                            features['energy'])
        tf.summary.histogram('HIST_ERROR', features['e_data'] - features['energy'])
        tf.summary.histogram('HIST_DATA', features['e_data'])
        tf.summary.histogram('HIST_PRED', features['energy'])

        return tf.estimator.EstimatorSpec(
            mode, loss=loss,
            train_op=optimizer.minimize(loss, global_step=global_step))

    if mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.losses.mean_squared_error(features['e_data'],
                                            features['energy'])

        metrics = {
            'MAE': tf.metrics.mean_absolute_error(
                features['e_data'], features['energy']),
            'RMSE': tf.metrics.root_mean_squared_error(
                features['e_data'], features['energy'])}

        return tf.estimator.EstimatorSpec(mode, loss=loss,
                                          eval_metric_ops=metrics)

    if mode == tf.estimator.ModeKeys.PREDICT:
        energy = features['energy']
        predictions = {
            'energy': energy,
            'forces': -tf.gradients(energy, features['coord'])[0]
        }
        print(predictions)
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)


def PiNN(model_dir='PiNN',
         depth=6, p_nodes=32, i_nodes=8, act='tanh', rc=4.0,
         atom_types=[1, 6, 7, 8], atomic_dress={0: 0.0}):
    """
    """
    filters = [
        f.atomic_mask(),
        f.atomic_dress(atomic_dress),
        f.distance_mat(),
        f.symm_func(),
        f.pi_basis(),
        f.pi_atomic(atom_types)
    ]

    layers = []

    for i in range(depth):
        layers += [
            l.fc_layer('fc_{}'.format(i), order=0, n_nodes=[p_nodes, p_nodes], act=act),
            l.pi_layer('pi_{}'.format(i), order=1, n_nodes=[p_nodes, i_nodes], act=act),
            l.ip_layer('ip_{}'.format(i), order=1, pool_type='sum'),
            l.en_layer('en_{}'.format(i), order=0, n_nodes=[p_nodes], act=act)
        ]

    params = {
        'filters': filters,
        'layers': layers,
        'dtype': tf.float32
    }

    estimator = tf.estimator.Estimator(
        model_fn=potential_model_fn, params=params, model_dir=model_dir)
    return estimator


def BPNN(model_dir='/tmp/BPNN',
         atomic_dress={0:0.0},
         elements=[1, 6, 7, 8],
         fc_depth=None,
         symm_funcs=None):
    """
    """
    if fc_depth is None:
        fc_depth = {i: [5,5,5] for i in elements}

    filters = [
        f.atomic_mask(),
        f.atomic_dress(atomic_dress),
        f.distance_mat(),
        f.symm_func()
    ]
    if symm_funcs is None:
        filters += [f.bp_G2(rs=rs) for rs in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]]
        filters += [f.bp_G3(lambd=lambd) for lambd in [0.8, 1.0, 1.2, 1.4]]
    else:
        filters += symm_funcs

    params = {
        'filters': filters,
        'layers': [l.bp_fc_layer(fc_depth)],
        'dtype': tf.float32
    }

    estimator = tf.estimator.Estimator(
        model_fn=potential_model_fn, params=params, model_dir=model_dir)
    return estimator

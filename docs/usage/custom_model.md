# Writing a PiNN model

## An Example 

Below is an simplified version of the potential model as a template to implement
new models. See also the API Documentation of the helper functions.

```Python
from pinn.models.base import export_model, get_train_op, MetricsCollector

@export_model
def simple_potential_model(features, labels, mode, params):
    """Model function for neural network potentials"""
    network = pinn.get_network(params['network'])
    model_params = default_params.copy()
    model_params.update(params['model_params'])

    features = network.preprocess(features)
    connect_dist_grad(features)
    pred = network(features)

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
        predictions = {'energy': pred}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

def make_metrics(features, pred, params, mode):
    metrics = MetricsCollector(mode)
    e_pred = pred
    e_data = features['e_data']
    e_mask = tf.abs(e_data) > params['max_energy'] if params['max_energy'] else None
    e_weight = params['e_loss_multiplier']
    e_weight *= features['e_weight'] if params['use_e_weight'] else 1
    metrics.add_error('E', e_data, e_pred, mask=e_mask, weight=e_weight,
                      use_error=(not params['use_e_per_atom']))
    return metrics
```

In the above code, the optimizer is defined by a model function, a more detailed
introduction to Estimators and model functions can be found in the TensorFlow 1
documentation.

The `MetricsCollector` object is a helper object in PiNN to handle different
forms of errors. It helps to apply customized weights to errors, filter them and
keep appropriate logs during the training and evaluation phases, see the API
documentation for more details.

# Writing a PiNN model

## Before you start

The `pinn.model` implements several . 

The documentation on writing custom Esitmators is removed in TF2 documentations.
We recommand reading the TensorFlow 1 version of the documentation.

## Example 

Below is an simplified version of the potential model as a template to implement
new models. See also the API Documentation of the helper functions.

```Python
@export_model('simple_potential_model')
def _simple_potential_model_fn(features, labels, mode, params):
    _check_params(params)
    network = params['network'](**params['network_params'])
    params = params['model_params']

    tensors = network.preprocess(features)
    pred = network(tensors)

    if mode == tf.estimator.ModeKeys.TRAIN:
        loss, error, metrics = get_loss(tensors, pred, params)
        make_train_summary(metrics)
        train_op = get_train_op(loss, error, params)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        loss, error, metrics = get_loss(tensors, pred, params)
        ops = make_eval_metric_ops(metrics)
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=ops)

    if mode == tf.estimator.ModeKeys.PREDICT:
        pred = pred / params['e_scale']
        pred += atomic_dress(features, params['e_dress']) if params['e_dress']
        predictions = {'energy': energy}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
```

The `_get_loss` function is the core of the model function, it should output
three variables:

- `loss`: The loss function to be used in the optimizer
- `error`: List of scaled error tensors
- `metrics`: List of metrics to be logged

!!! Note
    The errors list contains the scaled (by $\sqrt{w}$) error tensors. It is
    not used by default, and is reserved for the Extended Kalman Filter (EKF)
    optimizer.

```Python
def _get_loss(tensors, pred, params):
    loss, errors, metrics = 0, [], []
    e_pred = pred
    e_data = features['e_data']
    if model_params['e_dress']:
        e_data -= atomic_dress(features, model_params['e_dress'])
    e_data *= model_params['e_scale']

    # should get the mask here since max_energy refers to total energy
    e_mask = tf.abs(e_data) > params['max_energy'] if params['max_energy'] else False
    e_weight = params['e_loss_muiltiplier']
    e_weight *= features['e_weight'] if params['use_e_weight'] else 1
    add_error('E', e_data, e_pred, mask=e_mask, weight=e_weight)    
```

The `_check_params` function reads the parameters and checks for consistency.

```Python
def _check_params(params):
    assert params['network_params']
```

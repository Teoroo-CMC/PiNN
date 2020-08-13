# Optimizers in PiNN

The optimizer is an essential component in training neural networks. Since PiNN
networks are Keras models they can be optimized like any other Keras models, a
list of optimizers and their usage can be found in the TensorFlow
(documentation)[https://www.tensorflow.org/api_docs/python/tf/keras/optimizers]

## Using a optimizer in a PiNN model

All optimizers implemented in TensorFlow can be specified by the serialized
config. For instance, the default optimizer in PiNN can be specified with the
following block in the `params.yml` file.

```yaml
optimizer:
  class_name: Adam,
  config:
    learning_rate:
      class_name: ExponentialDecay,
      config:
        initial_learning_rate: 3e-4
        decay_steps: 10000
        decay_rate: 0.994
    clipnorm: 0.01
```


## extended Kalman Filter

The extended Kalman filter (EKF) is an alternative way to estimate the optimal
weights in neural networks. The EKF has long been used in combination with
atomic neural networks to approximate PES. EKF typically converges much faster
than a Stochastic Gradient Descent (SGD) based optimizer with more computation
cost per step.

Below is a recommended setup for using Kalman Filter in a PiNN model, which is
also available as `pinn.optimizers.default_kalman`

```yaml
optimizer:
  class_name: EKF,
  config:
    learning_rate:
      class_name: ExponentialDecay,
      config:
        initial_learning_rate: 0.03
        decay_steps: 10000
        decay_rate: 0.994
```

Note that the EKF implemented in PiNN is not a standard TensorFlow Optimizer
object, therefore, you can not use it directly as a regular optimizer, e.g.
`opt.minimize(loss)` in combination with the Keras models. This is mainly due to
the fact that `EFK` uses the error Jacobian rather than a scalar loss function
as input. Further updates might improve the API for better consistency.

import tensorflow as tf
from pinn.optimizers.ekf import EKF, default_ekf

default_adam = {
    'class_name': 'Adam',
    'config': {
        'learning_rate': {
            'class_name': 'ExponentialDecay',
            'config':{
                'initial_learning_rate': 3e-4,
                'decay_steps': 10000,
                'decay_rate': 0.994}},
        'clipnorm': 0.01}}

def get(optimizer):
    if isinstance(optimizer, EKF):
        return optimizer
    if isinstance(optimizer, dict) and optimizer['class_name']=='EKF':
        return EKF(**optimizer['config'])
    else:
        return tf.keras.optimizers.get(optimizer)


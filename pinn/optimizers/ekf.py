#!/usr/bin/env python
import tensorflow as tf

default_ekf = {
    'class_name': 'EKF',
    'config': {
        'learning_rate': 0.03}}

class EKF():
    """The EKF implementation follows mainly Singraber et al.'s description
    (Singraber, Morawietz, Behler and Dellage, JCTC, 2017), with some difference
    in the details about learning rate and noise scheduling.

    Args:
        learning_rate: learning rate
        inv_fp_prec (str): floating point precision for matrix inversion
        q_0: initial process noise
        q_tau: time constant for noise
        q_min: minimal noisee
    """
    def __init__(self, learning_rate, max_learning_rate=1.0,
                 epsilon=1.0, q_0=0.0, q_min=0.0, q_tau=3000.0,
                 inv_dtype='float64'):
        self.iterations = None
        self.learning_rate = learning_rate
        self.max_learning_rate = max_learning_rate
        self.epsilon = epsilon
        self.q_0 = q_0
        self.q_min = q_min
        self.q_tau = q_tau
        self.inv_dtype = tf.dtypes.as_dtype(inv_dtype)

    def get_train_op(self, error, tvars):
        from tensorflow.python.ops.parallel_for.gradients import jacobian
        from tensorflow.keras.optimizers.schedules import deserialize
        jacob = jacobian(error, tvars)
        HT = tf.concat([tf.reshape(j, [tf.shape(j)[0], -1]) for j in jacob], axis=1)
        H = tf.transpose(HT)
        m = tf.shape(H)[1]
        n = tf.reduce_sum(
            [tf.reduce_prod(var.shape) for var in tvars])
        tf.compat.v1.summary.scalar(f'KalmanFilter/m', m)
        tf.compat.v1.summary.scalar(f'KalmanFilter/n', n)
        P = tf.Variable(tf.eye(n, dtype=H.dtype)*self.epsilon, trainable=False)
        t = tf.cast(tf.compat.v1.train.get_global_step(), H.dtype)
        try:
            lr = deserialize(self.learning_rate)(t)
        except:
            lr = tf.cast(self.learning_rate, H.dtype)
        lr = tf.math.minimum(lr, self.max_learning_rate)
        # Computing Kalman Gain (avoid inversion, solve as linear equations)
        PH = tf.tensordot(P, H, 1)
        A_inv = tf.eye(m, dtype=H.dtype)/lr + tf.tensordot(HT, PH, 1)
        K = tf.linalg.lstsq(tf.cast(A_inv, self.inv_dtype),
                            tf.cast(tf.transpose(PH), self.inv_dtype),
                            fast=False)
        K = tf.transpose(tf.cast(K, H.dtype))
        grads = tf.tensordot(K, error, 1)
        lengths = [tf.reduce_prod(var.shape) for var in tvars]
        idx = tf.cumsum([0]+lengths)
        Q = tf.eye(n, dtype=H.dtype)*tf.math.maximum(tf.exp(-t/self.q_tau)*self.q_0, self.q_min)
        grads = [tf.reshape(grads[idx[i]:idx[i+1]], var.shape)
                 for i,  var in enumerate(tvars)]
        grads_and_vars = zip(grads, tvars)
        ops = [self.iterations.assign_add(1, read_value=False)]
        ops += [P.assign_add(Q-tf.tensordot(K, tf.transpose(PH),1), read_value=False)]
        ops += [var.assign_add(-grad, read_value=False) for grad, var in grads_and_vars]
        tf.compat.v1.summary.histogram(f'KalmanFilter/P_diag', tf.linalg.diag_part(P))
        tf.compat.v1.summary.histogram(f'KalmanFilter/P', P)
        train_op = tf.group(ops)
        return train_op

#!/usr/bin/env python
import tensorflow as tf

class gEKF():
    """
    gradient-based EKF, this should be equivalent with EKF, but faster

    Args:
        learning_rate: learning rate
        epsilon: scale initial guess for P matrix
        q_0: initial process noise
        q_tau: time constant for noise
        q_min: minimal noise
        div_prec (str): dtype for division
    """
    def __init__(self, learning_rate, epsilon=1,
                 q_0=0., q_min=0., q_tau=3000.0,
                 div_dtype='float64'):
        self.iterations = None
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.q_0 = q_0
        self.q_min = q_min
        self.q_tau = q_tau
        self.div_dtype = tf.dtypes.as_dtype(div_dtype)

    def get_train_op(self, error, tvars):
        from tensorflow.keras.optimizers.schedules import deserialize
        # gradients, initialize variables
        l1 = tf.reduce_mean(tf.abs(error))
        l2 = tf.reduce_mean(error**2)
        g1 = tf.gradients(l1, tvars)
        g2 = tf.gradients(l2, tvars)
        g1 = tf.concat([tf.reshape(g, [-1]) for g in g1], axis=0)
        g2 = tf.concat([tf.reshape(g, [-1]) for g in g2], axis=0)
        try:
            lr = deserialize(self.learning_rate)(t)
        except:
            lr = tf.cast(self.learning_rate, g1.dtype)
        # slots
        n = g1.shape[0]
        P = tf.Variable(tf.eye(n, dtype=g1.dtype)*self.epsilon, trainable=False)
        t = tf.cast(tf.compat.v1.train.get_global_step(), g1.dtype)
        lengths = [tf.reduce_prod(var.shape) for var in tvars]
        idx = tf.cumsum([0]+lengths)
        Pg1 = tf.einsum('ij,i->j', P, g1)
        Pg2 = tf.einsum('ij,i->j', P, g2)
        g1Pg1 = tf.einsum('i,i->', g1, Pg1)
        g1Pg2 = tf.einsum('i,i->', g1, Pg2)
        k = tf.cast(Pg1, self.div_dtype)/tf.cast(1./lr+g1Pg1, self.div_dtype)
        k = tf.cast(k, g2.dtype)
        grads = lr/2*(Pg2-k*g1Pg2) # preconditioned gradients for update
        tf.compat.v1.summary.histogram('KalmanFilter/updates', grads)
        Q = tf.eye(n, dtype=g1.dtype)*tf.math.maximum(tf.exp(-t/self.q_tau)*self.q_0, self.q_min)
        grads = [tf.reshape(grads[idx[i]:idx[i+1]], var.shape) for i,  var in enumerate(tvars)]
        grads_and_vars = zip(grads, tvars)
        ops = [self.iterations.assign_add(1, read_value=False)]
        ops += [P.assign_add(Q - tf.einsum('i,j->ij', k, Pg1), read_value=False)]
        ops += [var.assign_sub(grad, read_value=False) for grad, var in grads_and_vars]
        tf.compat.v1.summary.histogram('KalmanFilter/P_diag', tf.linalg.diag_part(P))
        tf.compat.v1.summary.histogram('KalmanFilter/P', P)
        train_op = tf.group(ops)
        return train_op

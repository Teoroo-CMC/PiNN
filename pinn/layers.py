"""
   Layers are operations on the dataset
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer


class fc_layer(object):
    """Documentation for fc_layer

    """

    def __init__(self, name, n_nodes, order=0, act='tanh'):
        self.n_nodes = n_nodes
        self.name = name
        self.order = order
        self.act = tf.nn.__getattribute__(act)

    def parse(self, tensors, dtype):
        nodes = tensors['nodes'][self.order]
        masks = tensors['pi_masks'][self.order]

        for i, n_out in enumerate(self.n_nodes):
            n_in = nodes.shape[-1]
            w = tf.get_variable('{}-w{}'.format(self.name, i),
                                shape=[n_in, n_out], dtype=dtype,
                                initializer=xavier_initializer())
            b = tf.get_variable('{}-b{}'.format(self.name, i),
                                shape=[1, n_out], dtype=dtype,
                                initializer=xavier_initializer())
            nodes = self.act(tf.tensordot(nodes, w, [-1, 0]) + b)
        nodes = masks * nodes
        tensors['nodes'][self.order] = nodes


class pi_layer(object):
    """Documentation for pi_layer

    """

    def __init__(self, name, n_nodes, order=1, act='tanh'):
        self.n_nodes = n_nodes
        self.name = name
        self.order = order
        self.act = tf.nn.__getattribute__(act)

    def parse(self, tensors, dtype):
        nodes = tensors['nodes'][self.order-1]
        masks = tensors['pi_masks'][self.order]
        i_kernel = tensors['pi_kernel'][self.order]

        # Shape: n x atoms x atoms x channel
        n_nodes = self.n_nodes.copy()
        n_kernel = i_kernel.shape[-1]
        n_nodes[-1] *= n_kernel

        nodes1 = tf.expand_dims(nodes, -2)
        nodes2 = tf.expand_dims(nodes, -3)
        nodes = tf.concat(
            [nodes1+tf.zeros_like(nodes2),
             nodes2+tf.zeros_like(nodes1)], -1)

        for i, n_out in enumerate(n_nodes):
            n_in = nodes.shape[-1]
            w = tf.get_variable('{}-w{}'.format(self.name, i),
                                shape=[n_in, n_out], dtype=dtype,
                                initializer=xavier_initializer())
            b = tf.get_variable('{}-b{}'.format(self.name, i),
                                shape=[n_out], dtype=dtype,
                                initializer=xavier_initializer())
            nodes = self.act(tf.tensordot(nodes, w, [-1, 0]) + b)

        # Shape: n x atoms x atoms x n_nodes x n_kernel
        nodes = tf.reshape(nodes,
                           tf.concat([tf.shape(nodes)[:-1],
                                      [self.n_nodes[-1], n_kernel]], 0))
        nodes = tf.reduce_sum(nodes * i_kernel, axis=-1)
        nodes = masks * nodes
        tensors['nodes'][self.order] = nodes


class ip_layer(object):
    """Documentation for ip_layer

    """

    def __init__(self, name,  order=1, pool_type='sum'):
        self.name = name
        self.order = order
        self.pool = {
            'sum': lambda x: tf.reduce_sum(x, axis=-2),
            'max': lambda x: tf.reduce_max(x, axis=-2)
        }[pool_type]

    def parse(self, tensors, dtype):
        i_nodes = tensors['nodes'][self.order]
        p_mask = tensors['pi_masks'][self.order - 1]

        p_nodes = self.pool(i_nodes)
        p_nodes = p_mask * p_nodes

        tensors['nodes'][self.order - 1] = p_nodes


class en_layer(object):
    """Documentation for fc_layer

    """

    def __init__(self, name, n_nodes, order=0, act='tanh'):
        self.n_nodes = n_nodes
        self.name = name
        self.order = order
        self.act = tf.nn.__getattribute__(act)

    def parse(self, tensors, dtype):
        nodes = tensors['nodes'][self.order]
        masks = tensors['pi_masks'][self.order]

        for i, n_out in enumerate(self.n_nodes):
            n_in = nodes.shape[-1]
            w = tf.get_variable('{}-w{}'.format(self.name, i),
                                shape=[n_in, n_out], dtype=dtype,
                                initializer=xavier_initializer())
            b = tf.get_variable('{}-b{}'.format(self.name, i),
                                shape=[1, n_out], dtype=dtype,
                                initializer=xavier_initializer())
            nodes = self.act(tf.tensordot(nodes, w, [-1, 0]) + b)

        nodes = masks * nodes

        w = tf.get_variable('{}-en'.format(self.name),
                            shape=[n_out], dtype=dtype,
                            initializer=xavier_initializer())

        nodes = tf.tensordot(nodes, w, [-1, 0])
        nodes = tf.reduce_sum(nodes, [i-1 for i in range(self.order + 1)])
        tensors['energy'] += nodes

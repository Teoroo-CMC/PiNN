import tensorflow as tf
import numpy as np


def get_variables(variables, dtype, shapes):
    tensors = []
    with tf.variable_scope('layers'):
        for i,var in enumerate(variables):
            trainable =  not ('fix' in var.keys() and var['fix'])
            if var['val'] is None:
                tensor = tf.get_variable(
                    dtype=dtype,
                    name=var['name'],
                    shape=shapes[i],
                    trainable=trainable,
                    initializer=tf.contrib.layers.xavier_initializer())
            else:
                tensor = tf.get_variable(
                    dtype=dtype,
                    name=var['name'],
                    shape=np.shape(var['val']),
                    trainable=trainable,
                    initializer=tf.constant_initializer(var['val']))
            tensors.append(tensor)
    return tensors[:]


class pinn_layer_base:
    ''' Template for layers of PiNN
    '''
    def __init__(self,
                 name,
                 n_nodes=10,
                 variables=None,
                 trainable=True,
                 activation='tanh',
                 collect_prop=False):
        self.name = name
        self.n_nodes = n_nodes
        self.variables = variables
        self.trainable = trainable
        self.activation = activation
        self.collect_prop = collect_prop

    def __repr__(self):
        return '{0}({1})'.format(self.name, self.n_nodes)

    def retrive_variables(self, sess):
        with tf.variable_scope('layers', reuse=True):
            for var in self.variables:
                var['val'] = sess.run(tf.get_variable(var['name'])).tolist()


class ip_layer(pinn_layer_base):
    '''Interaction pooling layer
    '''

    def __init__(self,
                 name,
                 n_nodes=8,
                 trainable=True,
                 activation='tanh',
                 collect_prop=False,
                 pool_type='max',
                 variables=None):
        pinn_layer_base.__init__(self, name, n_nodes, variables, trainable,
                                 activation, collect_prop)
        self.pool_type = pool_type
        if variables is None:
            variables = [{'name': '%s-w' % self.name, 'val':None},
                         {'name': '%s-b' % self.name, 'val':None}]
        self.variables = variables


    def process(self, i_nodes, p_nodes, i_mask, p_mask, i_in, p_output, dtype):
        shapes = [[i_nodes[-1].shape[-1], self.n_nodes],
                  [1, 1, self.n_nodes]]
        w, b = get_variables(self.variables, dtype ,shapes)
        act = tf.nn.__getattribute__(self.activation)
        output = act(tf.tensordot(i_nodes[-1], w, [[3], [0]]) + b)
        output = {
            'max': lambda x: tf.reduce_max(x, axis=-2),
            'sum': lambda x: tf.reduce_sum(x, axis=-2)
        }[self.pool_type](output)

        output = tf.where(tf.tile(p_mask, [1, 1, self.n_nodes]),
                          output, tf.zeros_like(output))
        p_nodes[-1] = tf.concat([p_nodes[-1], output], axis=-1)

        if self.collect_prop:
            p_output.append(output)


class pi_layer(pinn_layer_base):
    '''Interaction pooling layer
    '''

    def __init__(self,
                 name,
                 n_nodes=8,
                 trainable=True,
                 activation='tanh',
                 collect_prop=False,
                 variables=None):
        pinn_layer_base.__init__(self, name, n_nodes, variables, trainable,
                                 activation, collect_prop)
        if variables is None:
            variables = [{'name': '%s-w1' % self.name, 'val':None},
                         {'name': '%s-w2' % self.name, 'val':None},
                         {'name': '%s-b' % self.name, 'val':None}]
        self.variables = variables

    def process(self, i_nodes, p_nodes, i_mask, p_mask, i_in, p_output, dtype):

        shapes = [[p_nodes[-1].shape[-1], self.n_nodes],
                  [p_nodes[-1].shape[-1], self.n_nodes],
                  [1, 1, 1, self.n_nodes]]
        w1, w2, b = get_variables(self.variables, dtype, shapes)

        act = tf.nn.__getattribute__(self.activation)
        output = act(
            tf.expand_dims(tf.tensordot(p_nodes[-1], w1, [[2], [0]]), 1) +
            tf.expand_dims(tf.tensordot(p_nodes[-1], w2, [[2], [0]]), 2) + b)
        output = output * tf.tile(i_in, [1, 1, 1, self.n_nodes])
        output = tf.where(tf.tile(i_mask, [1, 1, 1, self.n_nodes]),
                          output, tf.zeros_like(output))
        i_nodes[-1] = tf.concat([i_nodes[-1], output], axis=-1)

        if self.collect_prop:
            p_output.append(output)


class pp_layer(pinn_layer_base):
    '''Interaction pooling layer
    '''

    def __init__(self,
                 name,
                 n_nodes=8,
                 trainable=True,
                 activation='tanh',
                 collect_prop=False,
                 variables=None):
        pinn_layer_base.__init__(self, name, n_nodes, variables, trainable,
                                 activation, collect_prop)
        if variables is None:
            variables = [{'name': '%s-w' % self.name, 'val':None},
                         {'name': '%s-b' % self.name, 'val':None}]
        self.variables = variables

    def process(self, i_nodes, p_nodes, i_mask, p_mask, i_in, p_output, dtype):
        shapes = [[p_nodes[-1].shape[-1], self.n_nodes],
                  [1, 1, self.n_nodes]]
        w, b = get_variables(self.variables, dtype, shapes)
        act = tf.nn.__getattribute__(self.activation)
        output = act(tf.tensordot(p_nodes[-1], w, [[2], [0]]) + b)
        output = tf.where(tf.tile(p_mask, [1, 1, self.n_nodes]),
                          output, tf.zeros_like(output))
        p_nodes.append(output)
        if self.collect_prop:
            p_output.append(output)


class ii_layer(pinn_layer_base):
    '''Interaction pooling layer
    '''

    def __init__(self,
                 name,
                 n_nodes=8,
                 trainable=True,
                 activation='tanh',
                 collect_prop=False,
                 variables=None):
        pinn_layer_base.__init__(self, name, n_nodes, variables, trainable,
                                 activation, collect_prop)
        if variables is None:
            variables = [{'name': '%s-w' % self.name, 'val':None},
                         {'name': '%s-b' % self.name, 'val':None}]
        self.variables = variables

    def process(self, i_nodes, p_nodes, i_mask, p_mask, i_in, p_output, dtype):
        shapes = [[i_nodes[-1].shape[-1], self.n_nodes],
                  [1, 1, 1, self.n_nodes]]
        w, b = get_variables(self.variables, dtype, shapes)
        act = tf.nn.__getattribute__(self.activation)
        output = act(tf.tensordot(i_nodes[-1], w, [[3], [0]]) + b)
        output = tf.where(tf.tile(i_mask, [1, 1, 1, self.n_nodes]),
                          output, tf.zeros_like(output))
        i_nodes.append(output)
        if self.collect_prop:
            p_output.append(output)


class e_layer(pinn_layer_base):
    '''Interaction pooling layer
    '''

    def __init__(self,
                 name='energy',
                 n_nodes=[16, 16],
                 trainable=True,
                 activation='tanh',
                 variables=None):
        pinn_layer_base.__init__(self, name, n_nodes, variables, trainable,
                                 activation)
        self.__delattr__('collect_prop')
        if variables is None:
            variables = []
            for i,n_node in enumerate(n_nodes):
                variables += [{'name': '%s-w%i' % (self.name, i), 'val':None},
                              {'name': '%s-b%i' % (self.name, i), 'val':None}]
            variables.append({'name':'%s-final' % self.name, 'val':None})
        self.variables = variables

    def process(self, i_nodes, p_nodes, i_mask, p_mask,
                i_in, p_output, dtype):
        output = tf.concat(p_output, axis=-1)
        shapes = []
        input_size = output.shape[-1]
        for i, n_node in enumerate(self.n_nodes):
            shapes += [[input_size, n_node],
                       [1, 1, n_node]]
            input_size = n_node
        shapes.append([input_size, 1])

        v = get_variables(self.variables, dtype, shapes)
        act = tf.nn.__getattribute__(self.activation)
        for i in range(len(self.n_nodes)):
            w, b = v[i*2], v[i*2+1]
            output = act(tf.tensordot(output, w, [[2], [0]]) + b)
            output = tf.where(tf.tile(p_mask, [1, 1, self.n_nodes[i]]),
                              output, tf.zeros_like(output))
        W = v[-1]
        output = tf.tensordot(output, W,  [[2], [0]])
        output = tf.where(p_mask, output, tf.zeros_like(output))
        energy = tf.reduce_sum(output, [1, 2])
        return energy


def default_layers(i_nodes=16, p_nodes=32, depth=1, activation='tanh'):
    layers = [
        pi_layer('pi-0', n_nodes=i_nodes),
        ip_layer('ip-0', n_nodes=p_nodes, collect_prop=True),
    ]
    for i in range(depth):
        layers += [
            ii_layer('ii-%i' % (i+1), n_nodes=i_nodes//2),
            pi_layer('pi-%i' % (i+1), n_nodes=i_nodes//2),
            pp_layer('pp-%i' % (i+1), n_nodes=p_nodes//2),
            ip_layer('ip-%i' % (i+1), n_nodes=p_nodes//4, pool_type='max'),
            ip_layer('ip-%i' % (i+1), n_nodes=p_nodes//4, pool_type='sum',
                     collect_prop=True)]
    layers.append(e_layer(n_nodes=[p_nodes]))
    return layers

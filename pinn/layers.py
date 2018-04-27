import tensorflow as tf
import numpy as np


def get_variables(variables, dtype, shapes):
    tensors = []
    with tf.variable_scope('layers'):
        for i, var in enumerate(variables):
            trainable = not ('fix' in var.keys() and var['fix'])
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
                 order=1,
                 n_nodes=10,
                 variables=None,
                 trainable=True,
                 activation='tanh'):
        self.name = name
        self.order = order
        self.n_nodes = n_nodes
        self.variables = variables
        self.trainable = trainable
        self.activation = activation

    def __repr__(self):
        return '{0}[{1},{2}]'.format(self.name, self.order, self.n_nodes)

    def retrive_variables(self, sess, dtype):
        with tf.variable_scope('layers', reuse=True):
            for var in self.variables:
                var['val'] = sess.run(tf.get_variable(var['name'],
                                                      dtype=dtype)).tolist()


class pi_compact(pinn_layer_base):
    '''Integrated PI and IP layer
    '''

    def __init__(self,
                 name,
                 order=1,
                 n_nodes=8,
                 trainable=True,
                 activation='tanh',
                 variables=None):
        pinn_layer_base.__init__(self, name, order, n_nodes, variables, trainable,
                                 activation)
        if variables is None:
            variables = [{'name': '%s-w1' % self.name, 'val': None},
                         {'name': '%s-w2' % self.name, 'val': None},
                         {'name': '%s-b' % self.name, 'val': None}]
        self.variables = variables

    def process(self, tensors, dtype):
        kernel = tensors['kernel']
        node = tensors['nodes'][self.order-1]
        mask = tensors['masks'][self.order]

        kernel = tf.expand_dims(kernel, axis=-1)

        for i in range(self.order-1):
            kernel = tf.expand_dims(kernel, axis=-3)
        shape = [1]*(self.order+2) + [self.n_nodes]
        shapes = [[node.shape[-1], kernel.shape[-2], self.n_nodes],
                  [node.shape[-1], kernel.shape[-2], self.n_nodes],
                  shape]
        w1, w2,  b = get_variables(self.variables, dtype, shapes)
        act = tf.nn.__getattribute__(self.activation)

        output = act(
            tf.expand_dims(tf.tensordot(node, w1, [[self.order+1], [0]]), 1) +
            tf.expand_dims(tf.tensordot(node, w2, [[self.order+1], [0]]), 2) + b)

        output = tf.reduce_sum(output * kernel, axis=-2)
        output = tf.where(tf.tile(mask, shape),
                          output, tf.zeros_like(output))
        slice = tf.abs(output[:,:,:,0:3])
        tf.summary.image(self.name, slice/tf.reduce_max(slice))
        output = tf.reduce_sum(output, axis=-2)

        tensors['nodes'][0] = tf.concat([tensors['nodes'][0], output], axis=-1)


class pi_flat(pinn_layer_base):
    '''Integrated PI and IP layer
    '''

    def __init__(self,
                 name,
                 order=1,
                 n_nodes=8,
                 trainable=True,
                 activation='tanh',
                 variables=None):
        pinn_layer_base.__init__(self, name, order, n_nodes, variables, trainable,
                                 activation)
        if variables is None:
            variables = []
            for i, n_node in enumerate(n_nodes):
                variables += [{'name': '%s-w%i' % (self.name, i), 'val': None},
                              {'name': '%s-b%i' % (self.name, i), 'val': None}]
        self.variables = variables

    def process(self, tensors, dtype):
        kernel = tensors['kernel']
        node = tensors['nodes'][self.order-1]
        mask = tensors['masks'][self.order]

        n_nodes = self.n_nodes.copy()
        n_kernel = kernel.shape[-1]
        n_nodes[-1] *= int(n_kernel)

        n_atoms, batch_size = node.shape[-2], node.shape[0]

        tile1 = [1, 1, n_atoms, 1]
        tile2 = [1, n_atoms, 1, 1]

        output = tf.concat([tf.tile(tf.expand_dims(node, -2), tile1),
                            tf.tile(tf.expand_dims(node, -3), tile2)],
                           axis=-1)
        input_size = output.shape[-1]
        shapes = []
        for i, n_node in enumerate(n_nodes):
            shapes += [[input_size, n_node],
                       [1, 1, 1, n_node]]
            input_size = n_node
        shapes.append([input_size, 1])

        v = get_variables(self.variables, dtype, shapes)
        act = tf.nn.__getattribute__(self.activation)

        for i in range(len(n_nodes)):
            w, b = v[i*2], v[i*2+1]
            output = act(tf.tensordot(output, w, [[3], [0]]) + b)

        output = tf.reshape(output,
                            [batch_size, n_atoms, n_atoms,
                             n_kernel, self.n_nodes[-1]])

        kernel = tf.expand_dims(kernel, -1)
        output = tf.reduce_sum(output * kernel, axis=-2)
        output = tf.where(tf.tile(mask, [1, 1, 1, self.n_nodes[-1]]),
                          output, tf.zeros_like(output))

        slice = tf.abs(output[:,:,:,0:3])
        tf.summary.image(self.name, slice/tf.reduce_max(slice))
        output = tf.reduce_sum(output, axis=-2)

        tensors['nodes'][0] = tf.concat([tensors['nodes'][0], output], axis=-1)


class ip_layer(pinn_layer_base):
    '''Interaction pooling layer
    '''

    def __init__(self,
                 name,
                 order=1,
                 n_nodes=8,
                 trainable=True,
                 activation='tanh',
                 pool_type='sum',
                 variables=None):
        pinn_layer_base.__init__(self, name, order, n_nodes, variables,
                                 trainable, activation)
        self.pool_type = pool_type
        if variables is None:
            variables = [{'name': '%s-w' % self.name, 'val': None},
                         {'name': '%s-b' % self.name, 'val': None}]
        self.variables = variables

    def process(self, tensors, dtype):
        node = tensors['nodes'][self.order]
        mask = tensors['masks'][self.order]

        shape = [1]*(self.order+2) + [self.n_nodes]
        shapes = [[node.shape[-1], self.n_nodes],
                  shape]
        w, b = get_variables(self.variables, dtype, shapes)
        act = tf.nn.__getattribute__(self.activation)

        output = act(tf.tensordot(node, w, [[self.order+2], [0]]) + b)
        output = tf.where(tf.tile(mask, shape),
                          output, tf.zeros_like(output))
        output = {
            'max': lambda x: tf.reduce_max(x, axis=-2),
            'sum': lambda x: tf.reduce_sum(x, axis=-2)
        }[self.pool_type](output)

        tensors['nodes'][self.order-1] = tf.concat([tensors['nodes'][self.order-1], output], axis=-1)


class pi_layer(pinn_layer_base):
    '''Pairwise interaction layer
    '''

    def __init__(self,
                 name,
                 order=1,
                 n_nodes=8,
                 trainable=True,
                 activation='tanh',
                 variables=None):
        pinn_layer_base.__init__(self, name, order, n_nodes, variables, trainable,
                                 activation)
        if variables is None:
            variables = [{'name': '%s-w1' % self.name, 'val': None},
                         {'name': '%s-w2' % self.name, 'val': None},
                         {'name': '%s-b' % self.name, 'val': None}]
        self.variables = variables

    def process(self, tensors, dtype):
        kernel = tensors['kernel']
        node = tensors['nodes'][self.order-1]
        mask = tensors['masks'][self.order]

        kernel = tf.expand_dims(kernel, axis=-1)

        for i in range(self.order-1):
            kernel = tf.expand_dims(kernel, axis=-3)
        shape = [1]*(self.order+2) + [self.n_nodes]
        shapes = [[node.shape[-1], kernel.shape[-2], self.n_nodes],
                  [node.shape[-1], kernel.shape[-2], self.n_nodes],
                  shape]
        w1, w2,  b = get_variables(self.variables, dtype, shapes)
        act = tf.nn.__getattribute__(self.activation)

        output = act(
            tf.expand_dims(tf.tensordot(node, w1, [[self.order+1], [0]]), 1) +
            tf.expand_dims(tf.tensordot(node, w2, [[self.order+1], [0]]), 2) + b)

        output = tf.reduce_sum(output * kernel, axis=-2)
        output = tf.where(tf.tile(mask, shape),
                          output, tf.zeros_like(output))

        tensors['nodes'][self.order] = tf.concat([tensors['nodes'][self.order], output], axis=-1)


class fc_layer(pinn_layer_base):
    '''Fully connected property layer
    '''
    def __init__(self,
                 name,
                 order=0,
                 n_nodes=8,
                 trainable=True,
                 activation='tanh',
                 variables=None):
        pinn_layer_base.__init__(self, name, order, n_nodes, variables, trainable,
                                 activation)
        if variables is None:
            variables = [{'name': '%s-w' % self.name, 'val': None},
                         {'name': '%s-b' % self.name, 'val': None}]
        self.variables = variables

    def process(self, tensors, dtype):
        node = tensors['nodes'][self.order]
        mask = tensors['masks'][self.order]

        shape = [1]*(self.order+2) + [self.n_nodes]
        shapes = [[node.shape[-1], self.n_nodes],
                  shape]

        w, b = get_variables(self.variables, dtype, shapes)
        act = tf.nn.__getattribute__(self.activation)
        output = act(tf.tensordot(node, w, [[self.order+2], [0]]) + b)
        output = tf.where(tf.tile(mask, shape),
                          output, tf.zeros_like(output))
        tensors['nodes'][self.order] = output


class en_layer(pinn_layer_base):
    '''Energy generation layer
    '''

    def __init__(self,
                 name,
                 order=0,
                 n_nodes=[32],
                 trainable=True,
                 activation='tanh',
                 variables=None):
        pinn_layer_base.__init__(self, name, order, n_nodes, variables, trainable,
                                 activation)
        if variables is None:
            variables = []
            for i, n_node in enumerate(n_nodes):
                variables += [{'name': '%s-w%i' % (self.name, i), 'val': None},
                              {'name': '%s-b%i' % (self.name, i), 'val': None}]
            variables.append({'name': '%s-final' % self.name, 'val': None})
        self.variables = variables

    def process(self, tensors, dtype):
        node = tensors['nodes'][self.order]
        mask = tensors['masks'][self.order]

        output = node
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
            output = tf.where(tf.tile(mask, [1]*(self.order+2)+ [self.n_nodes[i]]),
                              output, tf.zeros_like(output))
        W = v[-1]
        output = tf.tensordot(output, W,  [[self.order+2], [0]])
        output = tf.where(mask, output, tf.zeros_like(output))
        energy = tf.reduce_sum(output, [1, 2])
        return energy


def default_layers(p_nodes=32, i_nodes=[32,4], depth=6, act='tanh'):
    layers = [
        pi_flat('pi0', order=1, n_nodes=i_nodes, activation=act),
        en_layer('en0', order=0, n_nodes=[p_nodes])
    ]
    for i in range(1, depth+1):
        layers += [
            fc_layer('pp%i'%i, order=0, n_nodes=p_nodes, activation=act),
            pi_flat('pi%i'%i, order=1, n_nodes=i_nodes, activation=act),
            en_layer('en%i'%i, order=0, n_nodes=[p_nodes])
        ]
    return layers

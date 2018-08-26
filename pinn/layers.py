"""
   Layers are operations on the dataset
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from pinn.filters import sparse_node



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
        sparse = nodes.sparse

        for i, n_out in enumerate(self.n_nodes):
            n_in = sparse.shape[-1]
            w = tf.get_variable('{}-w{}'.format(self.name, i),
                                shape=[n_in, n_out], dtype=dtype,
                                initializer=xavier_initializer())
            b = tf.get_variable('{}-b{}'.format(self.name, i),
                                shape=[1, n_out], dtype=dtype,
                                initializer=xavier_initializer())
            sparse = self.act(tf.tensordot(sparse, w, [-1, 0]) + b)

        tensors['nodes'][self.order] = nodes.new_node(sparse)


class pi_layer(object):
    """Documentation for pi_layer

    """

    def __init__(self, name, n_nodes, order=1, act='tanh'):
        self.n_nodes = n_nodes
        self.name = name
        self.order = order
        self.act = tf.nn.__getattribute__(act)

    def parse(self, tensors, dtype):
        p_nodes = tensors['nodes'][self.order-1].get_dense()
        basis = tensors['pi_basis']

        if self.order == 1:
            indices = basis.indices
            mask = basis.mask
            basis = basis.sparse
        # else:
        #     indices =

        indices_i = indices[:, 0:-1]
        indices_j = tf.concat([indices[:, 0:-2], indices[:, -1:]], -1)

        nodes_i = tf.gather_nd(p_nodes, indices_i)
        nodes_j = tf.gather_nd(p_nodes, indices_j)
        nodes = tf.concat([nodes_i, nodes_j], -1)
        # Shape: n x atoms x atoms x channel
        n_nodes = self.n_nodes.copy()
        n_basis = basis.shape[-1]
        n_nodes[-1] *= n_basis
        # Fully Connected Layers
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
                                      [self.n_nodes[-1], n_basis]], 0))
        nodes = nodes * basis
        nodes = tf.reduce_sum(nodes, axis=-1)
        tensors['nodes'][self.order] = sparse_node(mask=mask,
                                                   indices=indices,
                                                   sparse=nodes)


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
        p_nodes = tensors['nodes'][self.order-1]

        sparse = tf.gather_nd(self.pool(i_nodes.get_dense()), p_nodes.indices)
        tensors['nodes'][self.order-1] = p_nodes.new_node(sparse)


class en_layer(object):
    """Documentation for en_layer
    """
    def __init__(self, name, n_nodes, order=0, act='tanh'):
        self.n_nodes = n_nodes
        self.name = name
        self.order = order
        self.act = tf.nn.__getattribute__(act)

    def parse(self, tensors, dtype):
        nodes = tensors['nodes'][self.order]
        sparse = nodes.sparse


        for i, n_out in enumerate(self.n_nodes):
            n_in = sparse.shape[-1]
            w = tf.get_variable('{}-w{}'.format(self.name, i),
                                shape=[n_in, n_out], dtype=dtype,
                                initializer=xavier_initializer())
            b = tf.get_variable('{}-b{}'.format(self.name, i),
                                shape=[1, n_out], dtype=dtype,
                                initializer=xavier_initializer())
            sparse = self.act(tf.tensordot(sparse, w, [-1, 0]) + b)

        w = tf.get_variable('{}-en'.format(self.name),
                            shape=[n_out], dtype=dtype,
                            initializer=xavier_initializer())
        sparse = tf.tensordot(sparse, w, [-1, 0])
        sparse = tf.SparseTensor(nodes.indices, sparse, nodes.mask.shape)
        sparse = tf.sparse_reduce_sum(sparse,
                                      [-i-1 for i in range(self.order+1)])
        tensors['energy'] += sparse


class bp_fc_layer(object):
    """Documentation for bp_fc_layer
    """
    def __init__(self,
                 n_nodes,
                 act='tanh'):
        self.name = 'bp_fc_layer'
        self.n_nodes = n_nodes
        self.act = tf.nn.__getattribute__(act)

    def parse(self, tensors, dtype):
        atoms = tensors['atoms']
        bp_sf = tensors['bp_sf']

        elem_maps = {}
        energy = 0
        for elem, n_nodes in self.n_nodes.items():
            elem_map = tf.gather_nd(atoms.indices,
                                    tf.where(tf.equal(atoms.sparse, elem)))
            nodes = tf.gather_nd(bp_sf, elem_map)
            for i, n_out in enumerate(n_nodes):
                n_in = nodes.shape[-1]
                w = tf.get_variable('bp-{}-w{}'.format(elem, i),
                                    shape=[n_in, n_out], dtype=dtype,
                                    initializer=xavier_initializer())
                b = tf.get_variable('bp-{}-b{}'.format(elem, i),
                                    shape=[n_out], dtype=dtype,
                                    initializer=xavier_initializer())
                nodes = self.act(tf.tensordot(nodes, w, [-1, 0]) + b)
            w = tf.get_variable('bp-{}-en'.format(elem),
                                shape=[n_out, 1], dtype=dtype,
                                initializer=xavier_initializer())

            nodes =  tf.reduce_sum(tf.tensordot(nodes, w, [-1, 0]), axis=-1)
            e_elem = tf.SparseTensor(elem_map, nodes, atoms.mask.shape)
            energy += tf.sparse_reduce_sum(e_elem, axis=-1)

        tensors['energy'] = energy

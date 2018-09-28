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
            sparse = tf.layers.dense(sparse, n_out,
                                     activation=self.act,
                                     name='{}-{}'.format(self.name, i))

        tensors['nodes'][self.order] = nodes.new_nodes(sparse)


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
        sparse = tf.concat([nodes_i, nodes_j], -1)
        # Shape: n x atoms x atoms x channel
        n_nodes = self.n_nodes.copy()
        n_basis = basis.shape[-1]
        n_nodes[-1] *= n_basis

        # Fully Connected Layers
        for i, n_out in enumerate(n_nodes):
            sparse = tf.layers.dense(sparse, n_out,
                                     activation=self.act,
                                     name='{}-{}'.format(self.name,i))

        sparse = tf.reshape(sparse,
                            tf.concat([tf.shape(sparse)[:-1],
                                       [self.n_nodes[-1], n_basis]], 0))
        sparse = sparse * basis
        sparse = tf.reduce_sum(sparse, axis=-1)
        tensors['nodes'][self.order] = sparse_node(mask=mask,
                                                   indices=indices,
                                                   sparse=sparse)


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
        tensors['nodes'][self.order-1] = p_nodes.new_nodes(sparse)


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
            sparse = tf.layers.dense(sparse, n_out,
                                     activation=self.act,
                                     name='{}-{}'.format(self.name, i))

        sparse = tf.layers.dense(sparse, 1, use_bias=False,
                                 activation=None,
                                 name='{}-en'.format(self.name))
        sparse = tf.squeeze(sparse, -1)

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
            sparse = tf.gather_nd(bp_sf, elem_map)

            for i, n_out in enumerate(n_nodes):
                sparse = tf.layers.dense(sparse, n_out,
                                         activation=self.act,
                                         name='{}-{}-{}'.format(self.name, elem, i))

            sparse = tf.layers.dense(sparse, 1, use_bias=False,
                                     activation=None,
                                     name='{}-{}-en'.format(self.name, elem))
            sparse = tf.squeeze(sparse, -1)

            e_elem = tf.SparseTensor(elem_map, sparse, atoms.mask.shape)
            energy += tf.sparse_reduce_sum(e_elem, axis=-1)

        tensors['energy'] = energy

class res_fc_layer(object):
    def __init__(self, order=0, n_nodes=[10], act='tanh'):
        self.name = 'res_fc_layer'
        self.order = order
        self.n_nodes = n_nodes
        self.act = tf.nn.__getattribute__(act)

    def parse(self, tensors, dtype):
        nodes = tensors['nodes'][self.order]
        sparse = nodes.sparse

        for i, n_out in enumerate(self.n_nodes):
            sparse_t = tf.layers.dense(sparse, sparse.shape[-1],
                                       activation=self.act,
                                       name='{}-{}t'.format(self.name, i))

            sparse = tf.layers.dense(tf.concat([sparse_t, sparse], -1), n_out,
                                     activation=None,
                                     name='{}-{}'.format(self.name, i))
        tensors['nodes'][self.order] = nodes.new_nodes(sparse)


class HIP_inter_layer(object):
    def __init__(self, n_nodes=10, act='tanh'):
        self.name = 'HIP_inter_layer'
        self.n_nodes = n_nodes
        self.act = tf.nn.__getattribute__(act)

    def parse(self, tensors, dtype):
        nodes = tensors['nodes'][0]
        basis = tensors['pi_basis']
        tensors['nodes'][0] = nodes

        sparse = tf.tensordot(basis, v) * tf.expand_dims(nodes)
        sparse = self.act(tf.reduce_sum(sparse, -2))
        tensors['nodes'][0] = nodes.new_nodes(sparse)


class SchNet_inter_layer(object):
    """
    A SchNet interaction layer is a filter to pool atomic properties.
    SchNet interaction layer currently does not create higher dimension nodes.
    """
    def __init__(self, n_nodes=10):
        self.name = 'SchNet_inter_layer'
        self.n_nodes = n_nodes
        self.act = tf.nn.__getattribute__(act)

    def parse(self, tensors, dtype):
        nodes = tensors['nodes'][0]
        basis = tensors['pi_basis']

        for i, n_out in enumerate(self.n_nodes):
            n_in = sparse.shape[-1]

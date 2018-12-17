# -*- coding: utf-8 -*-
"""Layers are operations on the tensors.

Layers here should be pure functions.
"""
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as default_init


def pi_layer(ind, nodes, basis,
             n_nodes=[4, 4], name='pi_layer',
             act='tanh'):
    """PiNN style interaction layer

    Args:
        ind: indices of the interating pair
        nodes: feature nodes of order (n-1)
        n_nodes: number of nodes to use
            Note that the last element of n_nodes specifies the dimention of
            the fully connected network before applying the basis function.
            Dimension of the last node is [pairs*n_nodes[-1]*n_basis], the
            output is then summed with the basis to form the interaction nodes

    Returns:
        Feature nodes of order n
    """
    ind_i = ind[:,0]
    ind_j = ind[:,1]
    prop_i = tf.gather(nodes, ind_i)
    prop_j = tf.gather(nodes, ind_j)
    inter = tf.concat([prop_i, prop_j], axis=-1)

    n_nodes_iter = n_nodes.copy()
    n_basis = basis.shape[-1]
    n_nodes_iter[-1] *= n_basis

    inter = fc_layer(inter, n_nodes_iter, act=act, name=name)
    inter = tf.reshape(inter, tf.concat(
        [tf.shape(inter)[:-1], [n_nodes[-1]], [n_basis]],0))
    inter = tf.reduce_sum(inter*basis, axis=-1)
    return inter


def ip_layer(ind, nodes, n_prop,
             pool_type='sum', name='ip_layer'):
    """Interaction pooling layer

    Args:
        ind: indices of the interaction
        nodes: feature nodes of order n
        n_prop: number of n-1 elements to pool into
        pool_type (string): sum or max
        Todo:
            Implement max pooling

    Return:
        Feature nodes of order n-1
    """
    prop = tf.unsorted_segment_sum(nodes, ind[:,0], n_prop)
    return prop


def fc_layer(nodes,
             n_nodes=[4,4], name='fc_layer',
             act='tanh'):
    """Fully connected layer, just a shortcut for multiple dense layers

    Args:
        n_node (list): dimension of the layers
        act: activation function of the layers
        name: name of the layer

    Returns:
        Nodes after the fc layers
    """
    for i,n_out in  enumerate(n_nodes):
        nodes = tf.layers.dense(nodes, n_out, activation=act,
                                name='{}-{}'.format(name, i))
    return nodes

def en_layer(ind, nodes, n_batch, n_nodes,
             name='en_layer',
             act='tanh'):
    """Just like ip layer, but allow for with fc_nodes and coefficients

    Args:
        ind: indices of the interaction
        nodes: feature nodes of order n
        n_batch: number of samples in a batch
        n_nodes: fc_layers before adding
    Return:
        Feature nodes of order n-1
    """
    for i,n_out in  enumerate(n_nodes):
        nodes = tf.layers.dense(nodes, n_out, activation=act,
                                name='{}-{}'.format(name, i))
    nodes = tf.layers.dense(nodes, 1, use_bias=False,
                            activation=None,
                            name='{}-en'.format(name))
    nodes = tf.unsorted_segment_sum(nodes, ind[:,0], n_batch)
    return tf.squeeze(nodes)


# class bp_fc_layer(object):
#     """Documentation for bp_fc_layer
#     """
#     def __init__(self,
#                  n_nodes,
#                  act='tanh'):
#         self.name = 'bp_fc_layer'
#         self.n_nodes = n_nodes
#         self.act = tf.nn.__getattribute__(act)

#     def parse(self, tensors, dtype):
#         atoms = tensors['atoms']
#         bp_sf = tensors['bp_sf']

#         elem_maps = {}
#         energy = 0
#         for elem, n_nodes in self.n_nodes.items():
#             elem_map = tf.gather_nd(atoms.indices,
#                                     tf.where(tf.equal(atoms.sparse, elem)))
#             sparse = tf.gather_nd(bp_sf, elem_map)

#             for i, n_out in enumerate(n_nodes):
#                 sparse = tf.layers.dense(sparse, n_out,
#                                          activation=self.act,
#                                          name='{}-{}-{}'.format(self.name, elem, i))

#             sparse = tf.layers.dense(sparse, 1, use_bias=False,
#                                      activation=None,
#                                      name='{}-{}-en'.format(self.name, elem))
#             sparse = tf.squeeze(sparse, -1)

#             e_elem = tf.SparseTensor(elem_map, sparse, atoms.mask.shape)
#             energy += tf.sparse_reduce_sum(e_elem, axis=-1)

#         tensors['energy'] = energy

# class res_fc_layer(object):
#     def __init__(self, order=0, n_nodes=[10], act='tanh'):
#         self.name = 'res_fc_layer'
#         self.order = order
#         self.n_nodes = n_nodes
#         self.act = tf.nn.__getattribute__(act)

#     def parse(self, tensors, dtype):
#         nodes = tensors['nodes'][self.order]
#         sparse = nodes.sparse

#         for i, n_out in enumerate(self.n_nodes):
#             sparse_t = tf.layers.dense(sparse, sparse.shape[-1],
#                                        activation=self.act,
#                                        name='{}-{}t'.format(self.name, i))

#             sparse = tf.layers.dense(tf.concat([sparse_t, sparse], -1), n_out,
#                                      activation=None,
#                                      name='{}-{}'.format(self.name, i))
#         tensors['nodes'][self.order] = nodes.new_nodes(sparse)


# class hip_inter_layer(object):
#     def __init__(self, n_nodes=10, act='tanh'):
#         self.name = 'HIP_inter_layer'
#         self.n_nodes = n_nodes
#         self.act = tf.nn.__getattribute__(act)

#     def parse(self, tensors, dtype):
#         nodes = tensors['nodes'][0]
#         basis = tensors['pi_basis']
#         tensors['nodes'][0] = nodes

#         sparse = tf.tensordot(basis, v) * tf.expand_dims(nodes)
#         sparse = self.act(tf.reduce_sum(sparse, -2))
#         tensors['nodes'][0] = nodes.new_nodes(sparse)


# class schnet_cfconv_layer(object):
#     """
#     A SchNet interaction layer is a filter to pool atomic properties.
#     SchNet interaction layer currently does not create higher dimension nodes.
#     """
#     def __init__(self, name='schnet_cfconv_layer',
#                  n_nodes=[64, 64], act='softplus'):
#         self.name = name
#         self.n_nodes = n_nodes
#         self.act = tf.nn.__getattribute__(act)

#     def parse(self, tensors, dtype):
#         nodes = tensors['nodes'][0]
#         basis = tensors['pi_basis']

#         indices_i = basis.indices[:, 0:-1]
#         nodes_i = tf.gather_nd(nodes.get_dense(), indices_i)
#         sparse = basis.sparse

#         for i, n_out in enumerate(self.n_nodes):
#             sparse = tf.layers.dense(sparse, n_out,
#                                      activation=self.act)

#         sparse = sparse * nodes_i
#         dense = sparse_node(mask=basis.mask,
#                             indices=basis.indices,
#                             sparse=sparse).get_dense()
#         tf.summary.image(self.name,
#                          dense[:,:,:,0:3])

#         sparse = tf.gather_nd(tf.reduce_sum(dense, axis=-3), nodes.indices)
#         sparse = tf.layers.dense(sparse, n_out, activation=None, use_bias=False)
#         sparse = sparse + nodes.sparse

#         tensors['nodes'][0] = nodes.new_nodes(sparse)

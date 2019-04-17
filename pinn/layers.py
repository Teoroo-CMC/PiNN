# -*- coding: utf-8 -*-
"""Layers are operations on the tensors.

Layers here should be pure functions.
"""
import numpy as np
import tensorflow as tf
from pinn.utils import pi_named
from tensorflow.contrib.layers import xavier_initializer as default_init

@pi_named('pi_layer')
def pi_layer(ind, nodes, basis,
             n_nodes=[4, 4],
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

    inter = fc_layer(inter, n_nodes_iter, act=act)
    inter = tf.reshape(inter, tf.concat(
        [tf.shape(inter)[:-1], [n_nodes[-1]], [n_basis]],0))
    inter = tf.reduce_sum(inter*basis, axis=-1)
    return inter

@pi_named('ip_layer')
def ip_layer(ind, nodes, n_prop,
             pool_type='sum'):
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

@pi_named('fc_layer')
def fc_layer(nodes,
             n_nodes=[4,4],
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
                                name='dense-{}'.format(i))
    return nodes


@pi_named('en_layer')
def en_layer(ind, nodes, n_batch, n_nodes,
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
                                name='dense-{}'.format(i))

    nodes = tf.layers.dense(nodes, 1, use_bias=False,
                            activation=None, name='energy')
    nodes = tf.unsorted_segment_sum(nodes, ind[:,0], n_batch)
    return tf.squeeze(nodes,-1)


def schnet_cfconv_layer():
    """ cfconv layer as described in 
    SchNet: https://doi.org/10.1063/1.5019779

    TODO: implement this
    """
    pass


def res_fc_layer():
    """ Fully connected layer with residue, as described in 
    HIPNN: https://doi.org/10.1063/1.5011181

    TODO: implement this
    """
    pass


def hip_ip_layer():
    """ HIPNN style interaction pooling layer

    TODO: implement this
    """
    pass


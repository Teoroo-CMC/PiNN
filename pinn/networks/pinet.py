# -*- coding: utf-8 -*-
import tensorflow as tf
from pinn.utils import pi_named, connect_dist_grad, \
    make_basis_jacob, connect_basis_jacob
from pinn.layers import cell_list_nl, cutoff_func, \
    polynomial_basis, gaussian_basis, atomic_onehot


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
    ind_i = ind[:, 0]
    ind_j = ind[:, 1]
    prop_i = tf.gather(nodes, ind_i)
    prop_j = tf.gather(nodes, ind_j)
    inter = tf.concat([prop_i, prop_j], axis=-1)

    n_nodes_iter = n_nodes.copy()
    n_basis = basis.shape[-1]
    n_nodes_iter[-1] *= n_basis

    inter = fc_layer(inter, n_nodes_iter, act=act)
    inter = tf.reshape(inter, tf.concat(
        [tf.shape(inter)[:-1], [n_nodes[-1]], [n_basis]], 0))
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
    prop = tf.unsorted_segment_sum(nodes, ind[:, 0], n_prop)
    return prop


@pi_named('fc_layer')
def fc_layer(nodes,
             n_nodes=[4, 4],
             act='tanh', use_bias=True):
    """Fully connected layer, just a shortcut for multiple dense layers

    Args:
        n_node (list): dimension of the layers
        act: activation function of the layers
        name: name of the layer

    Returns:
        Nodes after the fc layers
    """
    for i, n_out in enumerate(n_nodes):
        nodes = tf.layers.dense(nodes, n_out, activation=act, use_bias=use_bias,
                                name='dense-{}'.format(i))
    return nodes


@pi_named('en_layer')
def en_layer(nodes, n_nodes, act='tanh'):
    """Just like ip layer, but allow for with fc_nodes and coefficients

    Args:
        nodes: feature nodes of order n
        n_nodes: fc_layers before adding

    Returns:
        atomic prediction with shape (natoms)
    """
    for i, n_out in enumerate(n_nodes):
        nodes = tf.layers.dense(nodes, n_out, activation=act,
                                name='dense-{}'.format(i))

    nodes = tf.layers.dense(nodes, 1, use_bias=False,
                            activation=None, name='E_OUT')

    return tf.squeeze(nodes, -1)


def pinet(tensors, pp_nodes=[16, 16], pi_nodes=[16, 16],
          ii_nodes=[16, 16], en_nodes=[16, 16], depth=4,
          atom_types=[1, 6, 7, 8],  act='tanh',
          rc=4.0, cutoff_type='f1',
          basis_type='polynomial', n_basis=4, gamma=3.0,
          preprocess=False):
    """Network function for the PiNet neural network

    Args:
        tensors: input data (nested tensor from dataset).
        atom_types (list): elements for the one-hot embedding.
        pp_nodes (list): number of nodes for pp layer.
        pi_nodes (list): number of nodes for pi layer.
        ii_nodes (list): number of nodes for ii layer.
        en_nodes (list): number of nodes for en layer.
        depth (int): number of interaction blocks.
        rc (float): cutoff radius.
        basis_type (string): type of basis function to use,
            can be "polynomial" or "gaussian".
        gamma (float): controlles width of gaussian function for gaussian basis
        n_basis (int): number of basis functions to use.
        cutoff_type (string): cutoff function to use with the basis.
        act (string): activation function to use.
        preprocess (bool): whether to return the preprocessed tensor.

    Returns:
        prediction or preprocessed tensor dictionary
    """
    if "ind_2" not in tensors:
        tensors.update(cell_list_nl(tensors, rc))
        connect_dist_grad(tensors)
        tensors['embed'] = atomic_onehot(tensors['elems'], atom_types)
        if preprocess:
            return tensors
    else:
        connect_dist_grad(tensors)
    # Basis function
    if basis_type == 'polynomial':
        basis = polynomial_basis(tensors['dist'], cutoff_type, rc, n_basis)
    elif basis_type == 'gaussian':
        basis = gaussian_basis(
            tensors['dist'], cutoff_type, rc, n_basis, gamma)

    # We name some tensors here
    nodes = {1: tf.identity(tensors['embed'], name='embed')}
    diff = tf.identity(tensors['diff'], name='diff')
    coord = tf.identity(tensors['coord'], name='coord')
    elems = tf.identity(tensors['elems'], name='elems')
    ind_1 = tf.identity(tensors['ind_1'], name='ind_1')
    ind_2 = tf.identity(tensors['ind_2'], name='ind_2')
    basis = tf.expand_dims(basis, -2)
    natom = tf.shape(tensors['ind_1'])[0]
    # Then Construct the model
    output = 0.0
    for i in range(depth):
        if i > 0:
            nodes[1] = fc_layer(nodes[1], pp_nodes, act=act,
                                name='pp-{}'.format(i))
        nodes[2] = pi_layer(ind_2, nodes[1], basis, pi_nodes,
                            act=act, name='pi-{}'.format(i))
        nodes[2] = fc_layer(nodes[2], ii_nodes, use_bias=False,
                            act=act, name='ii-{}'.format(i))
        if nodes[1].shape[-1] != nodes[2].shape[-1]:
            nodes[1] = tf.layers.dense(nodes[1], nodes[2].shape[-1],
                                       use_bias=False, activation=None)
        nodes[1] = tf.add(
            nodes[1], ip_layer(ind_2, nodes[2], natom, name='ip_{}'.format(i)),
            name='prop_{}'.format(i))
        output = tf.add(
            output, en_layer(nodes[1], en_nodes, act=act,
                             name='en_{}'.format(i)),
            name='out_{}'.format(i))

    return output

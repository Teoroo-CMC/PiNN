# -*- coding: utf-8 -*-
"""Networks defines the structure of a model

Networks should be pure functions except during pre-processing, while
a nested tensors gets updated as input. Networks does not define the
goals/loss of the model, this allows for the usage of same network
structure for different tasks.
"""

import tensorflow as tf
import pinn.filters as f
import pinn.layers as l

def lj(tensors, rc=3.0, sigma=1.0, epsilon=1.0):
    """Lennard-Jones Potential

    Args:
        tensors: input data (nested tensor from dataset).
        rc: cutoff radius.
        sigma: 
        epsilon:
    """
    f.sparsify()(tensors)
    f.cell_list_nl(rc)(tensors)
    _reset_dist_grad(tensors)
    e0 = 4 * epsilon * ((sigma / rc)**12 - (sigma / rc)**6)
    c6 = (sigma/tensors['dist'])**6
    c12 = c6 ** 2
    en = 4*epsilon*(c12-c6)-e0
    ind = tensors['ind']
    natom = tf.shape(ind[1])[0]
    nbatch = tf.shape(tensors['atoms'])[0]
    en = tf.unsorted_segment_sum(en, ind[2][:,0], natom)
    en = tf.unsorted_segment_sum(en, ind[1][:,0], nbatch)
    return en/2.0

def pinn_network(tensors, pp_nodes=[16,16], pi_nodes=[16,16],
                 ii_nodes=[16,16], en_nodes=[16,16], depth=4,
                 atomic_dress={}, atom_types=[1,6,7,8],
                 rc=4.0, sf_type='f1', n_basis=4, act='tanh',
                 pre_level=0, preprocess=False,
                 to_return=0):
    """
    Args:
        tensors: input data (nested tensor from dataset).
        pp_nodes: number of nodes for pp layer.
        pi_nodes: number of nodes for pi layer.
        ii_nodes: number of nodes for ii layer.
        en_nodes: number of nodes for en layer.
        depth: number of interaction blocks.
        atom_types: number of types for the one hot encoding.
        rc: cutoff radius.
        sf_type: symmetry function to use with the basis.
        n_basis: number of polynomials to use with the basis.
        pre_level (int): flag for preprocessing:
            0: no preprocessing.
            1: preprocess till the cell list nl
            2: preprocess all filters (cannot do force training)
    Returns:
        - prediction tensor if n>=0
        - preprocessed nested tensors if n<0
    """
    filters = [
        f.sparsify(),
        f.atomic_onehot(atom_types),
        f.atomic_dress(atomic_dress),
        f.cell_list_nl(rc),
        f.symm_func(sf_type, rc),
        f.pi_basis(n_basis)]
    # Preprocess
    to_pre = {0: 0, 1: 4, 2: 6}[pre_level]
    if preprocess:
        for fi in filters[:to_pre]:
            fi(tensors)
        return tensors
    if pre_level==0:
        for fi in filters[:4]:
            fi(tensors)
    if pre_level<=1:
        _reset_dist_grad(tensors)
        for fi in filters[4:]:
            fi(tensors)
    # Then Construct the model
    nodes = {1: tensors['elem_onehot']}
    ind = tensors['ind']
    basis = tensors['pi_basis']
    natom = tf.shape(ind[1])[0]
    nbatch =  tf.shape(tensors['atoms'])[0]
    nodes[0] = 0.0
    for i in range(depth):
        if i>0:
            nodes[1] = l.fc_layer(nodes[1], pp_nodes,
                                  act=act, name='pp-{}/'.format(i))
        nodes[2] = l.pi_layer(ind[2], nodes[1], basis, pi_nodes,
                              act=act, name='pi-{}/'.format(i))
        nodes[2] = l.fc_layer(nodes[2], ii_nodes,
                              act=act, name='ii-{}/'.format(i))
        nodes[2] = nodes[2] * tensors['pi_basis'][:,:,0]
        nodes[1] = l.ip_layer(ind[2], nodes[2], natom,
                              name='ip_{}/'.format(i))
        nodes[0] += l.en_layer(ind[1], nodes[1], nbatch, en_nodes,
                               act=act, name='en_{}/'.format(i))
    return nodes[to_return]


def schnet_network(tensors):
    """ Network function for
    SchNet: https://doi.org/10.1063/1.5019779

    TODO: Implement this
    """
    pass

def bpnn_network(tensors):
    """ Network function for
    BPNN: https://doi.org/10.1103/PhysRevLett.98.146401

    TODO: Implement this
    """
    pass
# Helper functions
def _reset_dist_grad(tensors):
    tensors['diff'] = _connect_diff_grad(tensors['coord'], tensors['diff'],
                                         tensors['ind'][2])
    tensors['dist'] = _connect_dist_grad(tensors['diff'], tensors['dist'])

@tf.custom_gradient
def _connect_diff_grad(coord, diff, ind):
    """Returns a new diff with its gradients connected to coord"""
    def _grad(ddiff, coord, diff, ind):
        natoms = tf.shape(coord)[0]
        dcoord = tf.unsorted_segment_sum(ddiff, ind[:,1], natoms)
        dcoord -= tf.unsorted_segment_sum(ddiff, ind[:,0], natoms)
        return dcoord, None, None
    return tf.identity(diff), lambda ddiff: _grad(ddiff, coord, diff, ind)


@tf.custom_gradient
def _connect_dist_grad(diff, dist):
    """Returns a new dist with its gradients connected to diff"""
    def _grad(ddist, diff, dist):
        return tf.expand_dims(ddist/dist, 1)*diff, None
    return tf.identity(dist), lambda ddist: _grad(ddist, diff, dist)    

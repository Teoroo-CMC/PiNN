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

    This is a simple implementation of LJ potential
    for the purpose of testing distance/force calculations
    rather than a network.

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


def pinn_network(tensors, pp_nodes=[16, 16], pi_nodes=[16, 16],
                 ii_nodes=[16, 16], en_nodes=[16, 16], depth=4,
                 atomic_dress={}, atom_types=[1, 6, 7, 8],
                 rc=4.0, cutoff='f1', n_basis=4, act='tanh',
                 pre_level=0, preprocess=False,
                 to_return=0):
    """
    Args:
        tensors: input data (nested tensor from dataset).
        atom_types (list): elements for the one-hot embedding.
        pp_nodes (list): number of nodes for pp layer.
        pi_nodes (list): number of nodes for pi layer.
        ii_nodes (list): number of nodes for ii layer.
        en_nodes (list): number of nodes for en layer.
        depth (int): number of interaction blocks.
        rc (float): cutoff radius.
        n_basis (int): number of polynomials to use with the basis.
        cutoff (string): cutoff function to use with the basis.
        pre_level (int): flag for preprocessing:
            0 for no preprocessing;
            1 for preprocess till the cell list nl;
            2 for preprocess all filters (cannot do force training).
    Returns:
        - prediction tensor if n>=0
        - preprocessed nested tensors if n<0
    """
    filters = [
        f.sparsify(),
        f.atomic_onehot(atom_types),
        f.atomic_dress(atomic_dress),
        f.cell_list_nl(rc),
        f.cutoff_func(cutoff, rc),
        f.pi_basis(n_basis)]
    # Preprocess
    to_pre = {0: 0, 1: 4, 2: 6}[pre_level]
    if preprocess:
        for fi in filters[:to_pre]:
            fi(tensors)
        return tensors
    if pre_level == 0:
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
    nbatch = tf.shape(tensors['atoms'])[0]
    nodes[0] = 0.0
    for i in range(depth):
        if i > 0:
            nodes[1] = l.fc_layer(nodes[1], pp_nodes,
                                  act=act, name='pp-{}/'.format(i))
        nodes[2] = l.pi_layer(ind[2], nodes[1], basis, pi_nodes,
                              act=act, name='pi-{}/'.format(i))
        nodes[2] = l.fc_layer(nodes[2], ii_nodes,
                              act=act, name='ii-{}/'.format(i))
        nodes[2] = nodes[2] * tensors['pi_basis'][:, :, 0]
        nodes[1] = l.ip_layer(ind[2], nodes[2], natom,
                              name='ip_{}/'.format(i))
        nodes[0] += l.en_layer(ind[1], nodes[1], nbatch, en_nodes,
                               act=act, name='en_{}/'.format(i))
    return nodes[to_return]


def schnet_network(tensors):
    """ Network function for
    SchNet: https://doi.org/10.1063/1.5019779

    TODO: Implement this

    Args: 
        tensors: input data (nested tensor from dataset).
        gamma (float): "width" of the radial basis.
        miu_max (float): minimal distance of the radial basis.
        miu_min (float): maximal distance of the radial basis.
        n_basis (int): number of radial basis.
        n_atomic (int): number of nodes to be used in atomic layers.
        n_cfconv (int): number of nodes to be used in cfconv layers.
        T (int): number of interaction blocks.
        pre_level (int): flag for preprocessing:
            0 for no preprocessing;
            1 for preprocess till the cell list nl;
            2 for preprocess all filters (cannot do force training).
    Returns:
        - preprocessed nested tensors if n<0
        - prediction tensor if n>=0
    """
    pass


def bpnn_network(tensors, sf_spec, nn_spec, rc=5.0, act='tanh',
                 atomic_dress={}, preprocess=False, pre_level=0):
    """ Network function for Behler-Parrinello Neural Network

    Example of sf_spec:
        [{'type':'G2', 'i': 1, 'j': 8, 'Rs': [1.,2.], 'etta': [0.1,0.2]},
         {'type':'G2', 'i': 8, 'j': 1, 'Rs': [1.,2.], 'etta': [0.1,0.2]},
         {'type':'G4', 'i': 8, 'j': 8, 'lambd':[0.5,1], 'zeta': [1.,2.], 'etta': [0.1,0.2]}]

    The symmetry functions are defined according to this paper:
        Behler, Jörg. “Constructing High-Dimensional Neural Network Potentials: A Tutorial Review.” 
        International Journal of Quantum Chemistry 115, no. 16 (August 15, 2015): 103250. 
        https://doi.org/10.1002/qua.24890.
        (Note the naming of symmetry functiosn are different from http://dx.doi.org/10.1063/1.3553717)

    For more detials about symmetry functions, see the definitions of symmetry functions.

    Example of nn_spec:
        {8: [32, 32, 32],
         1: [16, 16, 16]}

    Args:
        tensors: input data (nested tensor from dataset).
        sf_spec (dict): symmetry function specification
        nn_spec (dict): elementwise network specification
            each key points to a list specifying the
            number of nodes in the fully-connected subnets.
        rc (float): cutoff radius.
        act (str): activation function to use in dense layers.
        preprocess (bool): 
            If True preprocess the input dataset instead of returning prediction.
        pre_level (int): flag for preprocessing:
            0 for no preprocessing;
            1 for preprocess till the cell list nl;
            2 for preprocess all filters (cannot do force training).
    Returns:
        - preprocessed nested tensors if preprocess
        - prediction tensor if not preprocess
    """
    filters = [
        f.sparsify(),
        f.atomic_dress(atomic_dress),
        f.cell_list_nl(rc),
        f.cutoff_func(rc=rc),
        f.bp_symm_func(sf_spec)]
    to_pre = {0: 0, 1: 3, 2: 5}[pre_level]
    if preprocess:
        for fi in filters[:to_pre]:
            fi(tensors)
        return tensors
    if pre_level == 0:
        for fi in filters[:4]:
            fi(tensors)
    if pre_level<=1:
        _reset_dist_grad(tensors)
        for fi in filters[4:]:
            fi(tensors)
    en = 0.0
    n_sample = tf.reduce_max(tensors['ind'][1])+1
    for k, v in nn_spec.items():
        with tf.name_scope("BP_DENSE_{}".format(k)):
            nodes = []
            ind = tf.where(tf.equal(tensors['elem'], k))        
            if k in tensors['symm_func']:
                nodes.append(tensors['symm_func'][k])
            if 'ALL' in tensors['symm_func']:
                nodes.append(tf.gather_nd(tensors['symm_func']['ALL'], ind))
            nodes = tf.concat(nodes, axis=-1)
            for n_node in v:
                nodes = tf.layers.dense(nodes, n_node, activation=act)
            atomic_en = tf.layers.dense(nodes, 1, activation=None, use_bias=False)
        en += tf.unsorted_segment_sum(
            atomic_en[:,0], tf.gather_nd(tensors['ind'][1], ind)[:,0], n_sample)
    return en

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
        dcoord = tf.unsorted_segment_sum(ddiff, ind[:, 1], natoms)
        dcoord -= tf.unsorted_segment_sum(ddiff, ind[:, 0], natoms)
        return dcoord, None, None
    return tf.identity(diff), lambda ddiff: _grad(ddiff, coord, diff, ind)


@tf.custom_gradient
def _connect_dist_grad(diff, dist):
    """Returns a new dist with its gradients connected to diff"""
    def _grad(ddist, diff, dist):
        return tf.expand_dims(ddist/dist, 1)*diff, None
    return tf.identity(dist), lambda ddist: _grad(ddist, diff, dist)    

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

def pinn_network(tensors, pp_nodes=[16,16], pi_nodes=[16,16],
                 ii_nodes=[16,16], en_nodes=[16,16], depth=4,
                 atomic_dress={}, atom_types=[1,6,7,8],
                 rc=4.0, sf_type='f1', n_basis=4,
                 pre_level=0):
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
            n: receiving inputs with n preprocessing.
            -n: excecute the first n preprocessing.
    Returns:
        - prediction tensor if n>=0
        - preprocessed nested tensors if n<0
    """
    filters = [
        f.sparsify(),
        f.atomic_onehot(atom_types),
        f.atomic_dress(atomic_dress),
        f.naive_nl(rc),
        f.symm_func(sf_type, rc),
        f.pi_basis(n_basis)]
    # Preprocess
    if pre_level<0:
        for fi in filters[:-pre_level]:
            fi(tensors)
        return tensors
    else:
        for fi in filters[pre_level:]:
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
            nodes[1] = l.fc_layer(nodes[1], pp_nodes, 'pp-{}'.format(i))
        nodes[2] = l.pi_layer(ind[2], nodes[1], basis, pi_nodes, 'pi-{}'.format(i))
        nodes[2] = l.fc_layer(nodes[2], ii_nodes, 'ii-{}'.format(i))
        nodes[1] = l.ip_layer(ind[2], nodes[2], natom, 'ip_{}'.format(i))
        nodes[0] += l.en_layer(ind[1], nodes[1], nbatch, en_nodes, 'en_{}'.format(i))
    return nodes[0]


# def schnet_network(tensors,
#            n_blockes=4, act='softplus', learning_rate=1e-4,
#            atom_types=[1, 6, 7, 8], atomic_dress={0: 0.0}):
#     """
#     """
#     filters = [
#         f.atomic_mask(),
#         f.atomic_dress(atomic_dress),
#         f.distance_mat(),
#         f.schnet_basis(),
#         f.pi_atomic(atom_types)
#     ]

#     layers = []
#     for i in range(n_blockes):
#         layers.append(l.fc_layer(n_nodes=[64], name='atom-wise-{}-1'.format(i)))
#         layers.append(l.schnet_cfconv_layer(name='cfconv-{}'.format(i)))
#         layers.append(l.fc_layer(n_nodes=[64], name='atom-wise-{}-2'.format(i)))

#     layers.append(l.fc_layer(n_nodes=[32], name='atom-wise-{}'.format(i+1)))
#     layers.append(l.en_layer('en_{}'.format(i), order=0, n_nodes=[32, 32], act=act))

#     params = {
#         'filters': filters,
#         'layers': layers,
#         'learning_rate': learning_rate,
#         'dtype': tf.float32
#     }

#     estimator = tf.estimator.Estimator(
#         model_fn=potential_model_fn, params=params,
#         model_dir=model_dir, config=config)
#     return estimator


# def BPNN(model_dir='/tmp/BPNN',
#          atomic_dress={0:0.0},
#          elements=[1, 6, 7, 8],
#          fc_depth=None,
#          learning_rate=1e-4,
#          symm_funcs=None):
#     """
#     """
#     if fc_depth is None:
#         fc_depth = {i: [5,5,5] for i in elements}

#     filters = [
#         f.atomic_mask(),
#         f.atomic_dress(atomic_dress),
#         f.distance_mat(),
#         f.symm_func()
#     ]

#     if symm_funcs is None:
#         filters += [f.bp_G2(rs=rs) for rs in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]]
#         filters += [f.bp_G3(lambd=lambd) for lambd in [0.8, 1.0, 1.2, 1.4]]
#     else:
#         filters += symm_funcs

#     params = {
#         'filters': filters,
#         'layers': [l.bp_fc_layer(fc_depth)],
#         'learning_rate': learning_rate,
#         'dtype': tf.float32
#     }

#     estimator = tf.estimator.Estimator(
#         model_fn=potential_model_fn, params=params, model_dir=model_dir)
#     return estimator


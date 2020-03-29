# -*- coding: utf-8 -*-
import tensorflow as tf
from pinn.utils import pi_named, connect_dist_grad
from pinn.layers import CellListNL, CutoffFunc, \
    PolynomialBasis, GaussianBasis, AtomicOnehot, ANNOutput

class FFLayer(tf.keras.layers.Layer):
    """Feed-forward layer, a shortcut for constructing multiple layers

    Args:
        n_node (list): dimension of the layers
        act: activation function of the layers
        name: name of the layer

    Returns:
        Nodes after the fc layers
    """
    def __init__(self, n_nodes=[64, 64], **kwargs):
        super(FFLayer, self).__init__()
        self.dense_layers = [tf.keras.layers.Dense(
            n_node, **kwargs) for n_node in n_nodes]

    def call(self, tensor):
        for layer in self.dense_layers:
            tensor = layer(tensor)
        return tensor

class PILayer(tf.keras.layers.Layer):
    """PiNN style interaction layer

    Args:
        n_nodes: number of nodes to use
            Note that the last element of n_nodes specifies the dimention of
            the fully connected network before applying the basis function.
            Dimension of the last node is [pairs*n_nodes[-1]*n_basis], the
            output is then summed with the basis to form the interaction nodes
        **kwargs: keyword arguments will be parsed to the feed forward layers
    """
    def __init__(self, n_nodes=[64], **kwargs):
        super(PILayer, self).__init__()
        self.n_nodes = n_nodes
        self.kwargs = kwargs

    def build(self, shapes):
        self.n_basis = shapes[2][-1]
        n_nodes_iter = self.n_nodes.copy()
        n_nodes_iter[-1] *= self.n_basis
        self.ff_layer = FFLayer(n_nodes_iter, **self.kwargs)

    def call(self, tensors):
        ind_2, prop, basis = tensors
        ind_i = ind_2[:, 0]
        ind_j = ind_2[:, 1]
        prop_i = tf.gather(prop, ind_i)
        prop_j = tf.gather(prop, ind_j)

        inter = tf.concat([prop_i, prop_j], axis=-1)
        inter = self.ff_layer(inter)
        inter = tf.reshape(inter, tf.concat(
            [tf.shape(inter)[:-1], [self.n_nodes[-1], self.n_basis]], 0))
        inter = tf.reduce_sum(inter*basis, axis=-1)
        return inter


class IPLayer(tf.keras.layers.Layer):
    """PiNet style IP layer

    transforms pairwise interactions to atomic properties"""
    def __init__(self):
        super(IPLayer, self).__init__()

    def call(self, tensors):
        ind_2, inter = tensors
        n_atoms = tf.reduce_max(ind_2) + 1
        return tf.math.unsorted_segment_sum(inter, ind_2[:, 0], n_atoms)


class OutLayer(tf.keras.layers.Layer):
    """PiNet style output layer

    generates outputs from atomic properties after each GC block"""

    def __init__(self, n_nodes, out_units, **kwargs):
        super(OutLayer, self).__init__()
        self.out_units = out_units
        self.ff_layer = FFLayer(n_nodes, **kwargs)
        self.out_units = tf.keras.layers.Dense(
            out_units, activation=None, use_bias=False)

    def call(self, tensors):
        ind_1, prop, prev_output = tensors
        prop = self.ff_layer(prop)
        output = self.out_units(prop) + prev_output
        return output

class GCBlock(tf.keras.layers.Layer):
    def __init__(self, pp_nodes, pi_nodes, ii_nodes, **kwargs):
        super(GCBlock, self).__init__()
        iiargs = kwargs.copy()
        iiargs.update(use_bias=False)
        self.pp_layer = FFLayer(pp_nodes, **kwargs)
        self.pi_layer = PILayer(pi_nodes, **kwargs)
        self.ii_layer = FFLayer(ii_nodes, **iiargs)
        self.ip_layer = IPLayer()

    def call(self, tensors):
        ind_2, prop, basis = tensors
        prop = self.pp_layer(prop)
        inter = self.pi_layer([ind_2, prop, basis])
        inter = self.ii_layer(inter)
        prop = self.ip_layer([ind_2, inter])
        return prop

class ResUpdate(tf.keras.layers.Layer):
    def __init__(self):
       super(ResUpdate, self).__init__()

    def build(self, shapes):
        assert isinstance(shapes, list) and len(shapes)==2
        if shapes[0][-1] == shapes[1][-1]:
            self.transform = lambda x:x
        else:
            self.transform = tf.keras.layers.Dense(
                shapes[1][-1], use_bias=False, activation=None)

    def call(self, tensors):
        old, new = tensors
        return self.transform(old) + new

class PreprocessLayer(tf.keras.layers.Layer):
    def __init__(self, atom_types, rc):
        super(PreprocessLayer, self).__init__()
        self.embed = AtomicOnehot(atom_types)
        self.nl_layer = CellListNL(rc)

    def call(self, tensors):
        tensors = tensors.copy()
        for k in ['elems', 'dist']:
            if k in tensors.keys():
                tensors[k] = tf.reshape(tensors[k], tf.shape(tensors[k])[:1])
        if 'ind_2' not in tensors:
            tensors.update(self.nl_layer(tensors))
            tensors['prop'] = self.embed(tensors['elems'])
        return tensors

class PiNet(tf.keras.Model):
    """Keras model for the PiNet neural network

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
        gamma (float): controls width of gaussian function for gaussian basis.
        n_basis (int): number of basis functions to use.
        cutoff_type (string): cutoff function to use with the basis.
        act (string): activation function to use.
        preprocess (bool): whether to return the preprocessed tensor.
    """
    def __init__(self, atom_types=[1, 6, 7, 8],  rc=4.0, cutoff_type='f1',
                 basis_type='polynomial', n_basis=4, gamma=3.0,
                 pp_nodes=[16, 16], pi_nodes=[16, 16], ii_nodes=[16, 16],
                 out_nodes=[16, 16], out_units=1, out_pool=False,
                 act='tanh', depth=4):

        super(PiNet, self).__init__()

        self.depth = depth
        self.preprocess = PreprocessLayer(atom_types, rc)

        if basis_type == 'polynomial':
            self.basis_fn = PolynomialBasis(cutoff_type, rc, n_basis)
        elif basis_type == 'gaussian':
            self.basis_fn = GaussianBasis(cutoff_type, rc, n_basis, gamma)

        self.res_update = [ResUpdate() for i in range(depth)]
        self.gc_blocks = [GCBlock([], pi_nodes, ii_nodes, activation=act)]
        self.gc_blocks += [GCBlock(pp_nodes, pi_nodes, ii_nodes, activation=act)
                           for i in range(depth-1)]
        self.out_layers = [OutLayer(out_nodes, out_units) for i in range(depth)]
        self.ann_output =  ANNOutput(out_pool)

    def call(self, tensors):
        tensors = self.preprocess(tensors)
        basis = self.basis_fn(tensors['dist'])[:, None, :]

        output = 0.0
        for i in range(self.depth):
            prop = self.gc_blocks[i]([tensors['ind_2'], tensors['prop'], basis])
            output = self.out_layers[i]([tensors['ind_1'], prop, output])
            tensors['prop'] = self.res_update[i]([tensors['prop'], prop])

        output = self.ann_output([tensors['ind_1'], output])
        return output

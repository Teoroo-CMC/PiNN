# -*- coding: utf-8 -*-

import tensorflow as tf
from pinn.layers import (
    CutoffFunc,
    PolynomialBasis,
    GaussianBasis,
    ANNOutput,
)

from .pinet import FFLayer, PILayer, IPLayer, ResUpdate
from .pinet2_p5_dot import PreprocessLayer, PIXLayer, AddLayer, ScaleLayer, OutLayer
    
class TensorProductLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        """
        Args:
            weighted (bool): style of the layer
        """
        super(TensorProductLayer, self).__init__()

    def call(self, tensor):
        """
        Args:
            tensor (`tensor`): tensor to be dot producted

        Returns:
            tensor: dot producted tensor
        """

        p3, p5 = tensor
        return tf.einsum('ijl,ijkl,ikl->il', p3, p5, p3)


class GCBlock(tf.keras.layers.Layer):
    def __init__(self, weighted: bool, pp_nodes, pi_nodes, ii_nodes, **kwargs):
        super(GCBlock, self).__init__()
        iiargs = kwargs.copy()
        iiargs.update(use_bias=False)
        ii1_nodes = ii_nodes.copy()
        ii1_nodes[-1] *= 4
        self.pp1_layer = FFLayer(pp_nodes, **kwargs)
        self.pi1_layer = PILayer(pi_nodes, **kwargs)
        self.ii1_layer = FFLayer(ii1_nodes, **iiargs)
        self.ip1_layer = IPLayer()

        self.pp3_layer = FFLayer(pp_nodes, activation=None, use_bias=False)
        self.pi3_layer = PIXLayer(weighted=weighted, **kwargs)
        self.ii3_layer = FFLayer(ii_nodes, **iiargs)
        self.ip3_layer = IPLayer()

        self.pp5_layer = FFLayer(pp_nodes, **kwargs)
        self.pi5_layer = PIXLayer(weighted=weighted, **kwargs)
        self.ii5_layer = FFLayer(ii_nodes, **iiargs)
        self.ip5_layer = IPLayer()
        self.add_layer = AddLayer()
        self.product_layer = TensorProductLayer()

        self.scale0_layer = ScaleLayer()
        self.scale1_layer = ScaleLayer()
        self.scale2_layer = ScaleLayer()
        self.scale3_layer = ScaleLayer()
        self.scale4_layer = ScaleLayer()

    def call(self, tensors):
        ind_2, p1, p3, p5, diff, basis = tensors
        p1 = self.pp1_layer(p1)
        i1 = self.pi1_layer([ind_2, p1, basis])
        i1 = self.ii1_layer(i1)
        i1_1, i1_2, i1_3, i1_4 = tf.split(i1, 4, axis=-1)
        p1 = self.ip1_layer([ind_2, p1, i1_1])
        scaled_diff = self.scale0_layer([diff[:, :, None], i1_2])

        p3 = self.pp3_layer(p3)
        i3 = self.pi3_layer([ind_2, p3])
        i3 = self.ii3_layer(i3)

        i3 = self.scale1_layer([i3, i1_3])
        i3 = i3 + scaled_diff
        p3 = self.ip3_layer([ind_2, p3, i3])

        p5 = self.pp5_layer(p5)
        i5 = self.pi5_layer([ind_2, p5])
        i5 = self.ii5_layer(i5)
        i5 = self.scale2_layer([i5, i1_4[:, None, :]])

        i5 = self.add_layer([i5, scaled_diff])
        p5 = self.ip5_layer([ind_2, p5, i5])
        p1t1 = self.product_layer([p3, p5]) + p1
        p3t1 = self.scale3_layer([p3, p1t1])
        p5t1 = self.scale4_layer([p5, p1t1[:, None, :]])

        return p1t1, p3t1, p5t1


class PiNet2P5Prod(tf.keras.Model):
    """This class implements the Keras Model for the PiNet network."""

    def __init__(
        self,
        atom_types=[1, 6, 7, 8],
        rc=4.0,
        cutoff_type="f1",
        basis_type="polynomial",
        n_basis=4,
        gamma=3.0,
        center=None,
        pp_nodes=[16, 16],
        pi_nodes=[16, 16],
        ii_nodes=[16, 16],
        out_nodes=[16, 16],
        out_units=1,
        out_pool=False,
        act="tanh",
        depth=4,
        weighted=True,
    ):
        """
        Args:
            atom_types (list): elements for the one-hot embedding
            pp_nodes (list): number of nodes for PPLayer
            pi_nodes (list): number of nodes for PILayer
            ii_nodes (list): number of nodes for IILayer
            out_nodes (list): number of nodes for OutLayer
            out_pool (str): pool atomic outputs, see ANNOutput
            depth (int): number of interaction blocks
            rc (float): cutoff radius
            basis_type (string): basis function, can be "polynomial" or "gaussian"
            n_basis (int): number of basis functions to use
            gamma (float or array): width of gaussian function for gaussian basis
            center (float or array): center of gaussian function for gaussian basis
            cutoff_type (string): cutoff function to use with the basis.
            act (string): activation function to use
            weighted (bool): whether to use weighted style
        """
        super(PiNet2P5Prod, self).__init__()

        self.depth = depth
        self.preprocess = PreprocessLayer(atom_types, rc)
        self.cutoff = CutoffFunc(rc, cutoff_type)

        if basis_type == "polynomial":
            self.basis_fn = PolynomialBasis(n_basis)
        elif basis_type == "gaussian":
            self.basis_fn = GaussianBasis(center, gamma, rc, n_basis)

        self.res_update1 = [ResUpdate() for i in range(depth)]
        self.res_update3 = [ResUpdate() for i in range(depth)]
        self.res_update5 = [ResUpdate() for i in range(depth)]
        self.gc_blocks = [GCBlock(weighted, [], pi_nodes, ii_nodes, activation=act)]
        self.gc_blocks += [
            GCBlock(weighted, pp_nodes, pi_nodes, ii_nodes, activation=act)
            for i in range(depth - 1)
        ]
        self.out_layers = [OutLayer(out_nodes, out_units) for i in range(depth)]
        self.ann_output = ANNOutput(out_pool)

    def call(self, tensors):
        """PiNet takes batches atomic data as input, the following keys are
        required in the input dictionary of tensors:

        - `ind_1`: [sparse indices](layers.md#sparse-indices) for the batched data, with shape `(n_atoms, 1)`;
        - `elems`: element (atomic numbers) for each atom, with shape `(n_atoms)`;
        - `coord`: coordintaes for each atom, with shape `(n_atoms, 3)`.

        Optionally, the input dataset can be processed with
        `PiNet.preprocess(tensors)`, which adds the following tensors to the
        dictionary:

        - `ind_2`: [sparse indices](layers.md#sparse-indices) for neighbour list, with shape `(n_pairs, 2)`;
        - `dist`: distances from the neighbour list, with shape `(n_pairs)`;
        - `diff`: distance vectors from the neighbour list, with shape `(n_pairs, 3)`;
        - `prop`: initial properties `(n_pairs, n_elems)`;

        Args:
            tensors (dict of tensors): input tensors

        Returns:
            output (tensor): output tensor with shape `[n_atoms, out_nodes]`
        """
        tensors = self.preprocess(tensors)
        tensors["p3"] = tf.zeros([tf.shape(tensors["ind_1"])[0], 3, 1])
        tensors["p5"] = tf.zeros([tf.shape(tensors["ind_1"])[0], 3, 3, 1])
        fc = self.cutoff(tensors["dist"])
        basis = self.basis_fn(tensors["dist"], fc=fc)
        output = 0.0
        for i in range(self.depth):
            p1, p3, p5 = self.gc_blocks[i](
                [tensors["ind_2"], tensors["p1"], tensors["p3"], tensors["p5"], tensors["norm_diff"], basis]
            )
            output = self.out_layers[i]([tensors["ind_1"], p1, p3, output])
            tensors["p1"] = self.res_update1[i]([tensors["p1"], p1])
            tensors["p3"] = self.res_update3[i]([tensors["p3"], p3])
            tensors["p5"] = self.res_update5[i]([tensors["p5"], p5])

        output = self.ann_output([tensors["ind_1"], output])
        return output

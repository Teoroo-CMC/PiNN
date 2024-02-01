# -*- coding: utf-8 -*-

import tensorflow as tf
from pinn.layers import (
    CellListNL,
    CutoffFunc,
    PolynomialBasis,
    GaussianBasis,
    AtomicOnehot,
    ANNOutput,
)

from .pinet import FFLayer, PILayer, IPLayer, ResUpdate


class PIXLayer(tf.keras.layers.Layer):
    R"""`PIXLayer` takes the equalvariant properties ${}^{3}\mathbb{P}_{ix\zeta}$ as input and outputs interactions for each pair ${}^{3}\mathbb{I}_{ijx\zeta}$. The `PIXLayer` has two styles, specified by the `weighted` argument:

    `weighted`:
    $$
    \begin{aligned}
    {}^{3}\mathbb{I}_{ijx\gamma} = W_{\zeta\gamma}^{'} \mathbf{1}_{j}^{'} {}^{3}\mathbb{P}_{ix\zeta} + W_{\zeta\gamma}^{''} \mathbf{1}_{i}^{''} {}^{3}\mathbb{P}_{jx\zeta}
    \end{aligned}
    $$


    `non-weighted`:
    $$
    \begin{aligned}
    {}^{3}\mathbb{I}_{ijx\zeta} = \mathbf{1}_{j} {}^{3}\mathbb{P}_{ix\zeta}
    \end{aligned}
    $$

    """

    def __init__(self, weighted: bool, **kwargs):
        """
        Args:
            weighted (bool): style of the layer, should be a bool
        """
        super(PIXLayer, self).__init__()
        self.weighted = weighted

    def build(self, shapes):
        if self.weighted:
            self.wi = tf.keras.layers.Dense(
                shapes[1][-1], activation=None, use_bias=False
            )
            self.wj = tf.keras.layers.Dense(
                shapes[1][-1], activation=None, use_bias=False
            )

    def call(self, tensors):
        """
        PILayer take a list of three tensors as input:

        - ind_2: [sparse indices](layers.md#sparse-indices) of pairs with shape `(n_pairs, 2)`
        - prop: equalvariant tensor with shape `(n_atoms, x, n_prop)`

        Args:
            tensors (list of tensors): list of `[ind_2, prop]` tensors

        Returns:
            inter (tensor): interaction tensor with shape `(n_pairs, x, n_nodes[-1])`
        """
        ind_2, px = tensors
        ind_i = ind_2[:, 0]
        ind_j = ind_2[:, 1]
        px_i = tf.gather(px, ind_i)
        px_j = tf.gather(px, ind_j)

        if self.weighted:
            return self.wi(px_i) + self.wj(px_j)
        else:
            return px_j


class DotLayer(tf.keras.layers.Layer):
    R"""`DotLayer` stands for the dot product( $\langle,\rangle$ ). `DotLayer` has two styles, specified by the `weighted` argument:

    `weighted`:

    $$
    \begin{aligned}
    {}^{3}\mathbb{P}_{i\zeta} = \sum_{x\alpha\beta} W_{\beta\zeta}^{'} W_{\alpha\zeta}^{''}  {}^{3}\mathbb{P}_{ix\alpha} {}^{3}\mathbb{P}_{ix\beta}
    \end{aligned}
    $$

    `non-weighted`:

    $$
    \begin{aligned}
    {}^{3}\mathbb{P}_{i\zeta} = \sum_x {}^{3}\mathbb{P}_{ix\zeta} {}^{3}\mathbb{P}_{ix\zeta}
    \end{aligned}
    $$
    """

    def __init__(self, weighted: bool, **kwargs):
        """
        Args:
            weighted (bool): style of the layer
        """
        super(DotLayer, self).__init__()
        self.weighted = weighted

    def build(self, shapes):
        if self.weighted:
            self.wi = tf.keras.layers.Dense(shapes[-1], activation=None, use_bias=False)
            self.wj = tf.keras.layers.Dense(shapes[-1], activation=None, use_bias=False)

    def call(self, tensor):
        """
        Args:
            tensor (`tensor`): tensor to be dot producted

        Returns:
            tensor: dot producted tensor
        """
        if self.weighted:
            return tf.einsum("ixr,ixr->ir", self.wi(tensor), self.wj(tensor))
        else:
            return tf.einsum("ixr,ixr->ir", tensor, tensor)


class ScaleLayer(tf.keras.layers.Layer):
    R"""`ScaleLayer` represents the scaling of a equalvariant property tensor by a scalar, and has no learnable variables. The `ScaleLayer` takes two tensors as input and outputs a tensor of the same shape as the first input tensor, i.e.:

    $$
    \begin{aligned}
    \mathbb{X}_{..x\alpha} = \mathbb{X}_{..x\alpha} \mathbb{X}_{..\alpha}
    \end{aligned}
    $$
    """

    def __init__(self, **kwargs):
        super(ScaleLayer, self).__init__()

    def __call__(self, tensor):
        """
        Args:
            tensor (list of tensors): list of `[tensor, scalar]` tensors

        Returns:
            tensor: scaled tensor
        """
        px, p1 = tensor
        return px * p1[:, None, :]


class OutLayer(tf.keras.layers.Layer):
    """`OutLayer` updates the network output with a `FFLayer` layer, where the
    `out_units` controls the dimension of outputs. In addition to the `FFLayer`
    specified by `n_nodes`, the `OutLayer` has one additional linear biasless
    layer that scales the outputs, specified by `out_units`.

    """

    def __init__(self, n_nodes, out_units, **kwargs):
        """
        Args:
            n_nodes (list): dimension of the hidden layers
            out_units (int): dimension of the output units
            **kwargs (dict): options to be parsed to dense layers
        """
        super(OutLayer, self).__init__()
        self.out_units = out_units
        self.ff_layer = FFLayer(n_nodes, **kwargs)
        self.out_units = tf.keras.layers.Dense(
            out_units, activation=None, use_bias=False
        )

    def call(self, tensors):
        """
        OutLayer takes a list of three tensors as input:

        - ind_1: [sparse indices](layers.md#sparse-indices) of atoms with shape `(n_atoms, 2)`
        - prop: property tensor with shape `(n_atoms, n_prop)`
        - prev_output:  previous output with shape `(n_atoms, out_units)`

        Args:
            tensors (list of tensors): list of [ind_1, prop, prev_output] tensors

        Returns:
            output (tensor): an updated output tensor with shape `(n_atoms, out_units)`
        """
        ind_1, p1, p3, prev_output = tensors
        p1 = self.ff_layer(p1)
        output = self.out_units(p1) + prev_output
        return output


class GCBlock(tf.keras.layers.Layer):
    def __init__(self, weighted: bool, pp_nodes, pi_nodes, ii_nodes, **kwargs):
        super(GCBlock, self).__init__()
        iiargs = kwargs.copy()
        iiargs.update(use_bias=False)
        ii_nodes = ii_nodes.copy()
        ii_nodes[-1] *= 3
        self.pp1_layer = FFLayer(pp_nodes, **kwargs)
        self.pi1_layer = PILayer(pi_nodes, **kwargs)
        self.ii1_layer = FFLayer(ii_nodes, **iiargs)
        self.ip1_layer = IPLayer()

        self.pp3_layer = FFLayer(pp_nodes, activation=None, use_bias=False)
        self.pix_layer = PIXLayer(weighted=weighted, **kwargs)
        self.ii3_layer = FFLayer(ii_nodes, **iiargs)
        self.ip3_layer = IPLayer()

        self.dot_layer = DotLayer(weighted=weighted)

        self.scale1_layer = ScaleLayer()
        self.scale2_layer = ScaleLayer()
        self.scale3_layer = ScaleLayer()

    def call(self, tensors):
        ind_2, p1, p3, diff, basis = tensors

        p1 = self.pp1_layer(p1)
        i1 = self.pi1_layer([ind_2, p1, basis])
        i1 = self.ii1_layer(i1)
        i1_1, i1_2, i1_3 = tf.split(i1, 3, axis=-1)
        p1 = self.ip1_layer([ind_2, p1, i1_2])

        p3 = self.pp3_layer(p3)
        i3 = self.pix_layer([ind_2, p3])
        i3 = self.scale1_layer([i3, i1_3])
        scaled_diff = self.scale2_layer([diff[:, :, None], i1_1])
        i3 = i3 + scaled_diff
        p3 = self.ip3_layer([ind_2, p3, i3])

        p1t1 = self.dot_layer(p3) + p1
        p3t1 = self.scale3_layer([p3, p1t1])

        return p1t1, p3t1


class PreprocessLayer(tf.keras.layers.Layer):
    def __init__(self, atom_types, rc):
        super(PreprocessLayer, self).__init__()
        self.embed = AtomicOnehot(atom_types)
        self.nl_layer = CellListNL(rc)

    def call(self, tensors):
        tensors = tensors.copy()
        for k in ["elems", "dist"]:
            if k in tensors.keys():
                tensors[k] = tf.reshape(tensors[k], tf.shape(tensors[k])[:1])
        if "ind_2" not in tensors:
            tensors.update(self.nl_layer(tensors))
            tensors["p1"] = tf.cast(  # difference with pinet: prop->p1
                self.embed(tensors["elems"]), tensors["coord"].dtype
            )
        return tensors


class PiNet2(tf.keras.Model):
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
        super(PiNet2, self).__init__()

        self.depth = depth
        self.preprocess = PreprocessLayer(atom_types, rc)
        self.cutoff = CutoffFunc(rc, cutoff_type)

        if basis_type == "polynomial":
            self.basis_fn = PolynomialBasis(n_basis)
        elif basis_type == "gaussian":
            self.basis_fn = GaussianBasis(center, gamma, rc, n_basis)

        self.res_update1 = [ResUpdate() for i in range(depth)]
        self.res_update3 = [ResUpdate() for i in range(depth)]
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
        fc = self.cutoff(tensors["dist"])
        basis = self.basis_fn(tensors["dist"], fc=fc)
        output = 0.0
        for i in range(self.depth):
            p1, p3 = self.gc_blocks[i](
                [tensors["ind_2"], tensors["p1"], tensors["p3"], tensors["diff"], basis]
            )
            output = self.out_layers[i]([tensors["ind_1"], p1, p3, output])
            tensors["p1"] = self.res_update1[i]([tensors["p1"], p1])
            tensors["p3"] = self.res_update3[i]([tensors["p3"], p3])

        output = self.ann_output([tensors["ind_1"], output])
        return output

# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from pinn.layers import (
    CellListNL,
    CutoffFunc,
    PolynomialBasis,
    GaussianBasis,
    AtomicOnehot,
    ANNOutput,
)

from pinn.networks.pinet import FFLayer, PILayer, IPLayer, ResUpdate

class PIXLayer(tf.keras.layers.Layer):
    r"""`PIXLayer` takes the equalvariant properties ${}^{3}\mathbb{P}_{ix\zeta}$ as input and outputs interactions for each pair ${}^{3}\mathbb{I}_{ijx\zeta}$. The `PIXLayer` has two styles, specified by the `weighted` argument. The weight matrix $W$ is initialized randomly using the `glorot_uniform` initializer.

    `non-weighted` (default):

    $$
    \begin{aligned}
    {}^{3}\mathbb{I}_{ijx\gamma} = \mathbf{1}_{j} {}^{3}\mathbb{P}_{ix\gamma}
    \end{aligned}
    $$

    $$
    \begin{aligned}
    {}^{5}\mathbb{I}_{ijxy\gamma} = \mathbf{1}_{j} {}^{5}\mathbb{P}_{ixy\gamma}
    \end{aligned}
    $$

    `weighted` (experimental):

    $$
    \begin{aligned}
    {}^{3}\mathbb{I}_{ijx\gamma} = W_{\gamma\gamma} \mathbf{1}_{j} {}^{3}\mathbb{P}_{ix\gamma} + W_{\gamma\gamma}^{'} \mathbf{1}_{i}^{'} {}^{3}\mathbb{P}_{jx\gamma}
    \end{aligned}
    $$

    $$
    \begin{aligned}
    {}^{5}\mathbb{I}_{ijxy\gamma} = W_{\gamma\gamma} \mathbf{1}_{j} {}^{5}\mathbb{P}_{ixy\gamma} + W_{\gamma\gamma}^{'} \mathbf{1}_{i}^{'} {}^{5}\mathbb{P}_{jxy\gamma}
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

    r"""`DotLayer` stands for the dot product $\langle,\rangle$. `DotLayer` has two styles, specified by the `weighted` argument. The weight matrix $W$ is initialized randomly using the `glorot_uniform` initializer.

    `non-weighted` (default):

    $$
    \begin{aligned}
    {}^{3}\mathbb{P}_{i\gamma} = \sum_x {}^{3}\mathbb{P}_{ix\gamma} {}^{3}\mathbb{P}_{ix\gamma}
    \end{aligned}
    $$

    $$
    \begin{aligned}
    {}^{5}\mathbb{P}_{i\gamma} = \sum_{xy} {}^{5}\mathbb{P}_{ixy\gamma} {}^{5}\mathbb{P}_{ixy\gamma}
    \end{aligned}
    $$

    `weighted` (experimental):

    $$
    \begin{aligned}
    {}^{3}\mathbb{P}_{i\gamma} = \sum_{x} W_{\gamma\gamma}   {}^{3}\mathbb{P}_{ix\gamma} W_{\gamma\gamma}^{'} {}^{3}\mathbb{P}_{ix\gamma}
    \end{aligned}
    $$

    $$
    \begin{aligned}
    {}^{5}\mathbb{P}_{i\gamma} = \sum_{xy} W_{\gamma\gamma}   {}^{5}\mathbb{P}_{ixy\gamma} W_{\gamma\gamma}^{'} {}^{5}\mathbb{P}_{ixy\gamma}
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
            tensor (`tensor`): dot producted tensor
        """
        if self.weighted:
            return tf.einsum("ixr,ixr->ir", self.wi(tensor), self.wj(tensor))
        else:
            return tf.einsum("ixr,ixr->ir", tensor, tensor)


class ScaleLayer(tf.keras.layers.Layer):
    r"""`ScaleLayer` represents the scaling of a equalvariant property tensor by a scalar, and has no learnable variables. The `ScaleLayer` takes two tensors as input and outputs a tensor of the same shape as the first input tensor, i.e.:

    $$
    \begin{aligned}
    \mathbb{X}_{..x\alpha}^{\prime} = \mathbb{X}_{..x\alpha} \mathbb{X}_{..\alpha}
    \end{aligned}
    $$
    """

    def __init__(self, **kwargs):
        super(ScaleLayer, self).__init__()

    def call(self, tensor):
        """
        Args:
            tensor (list of tensors): list of `[tensor, scalar]` tensors

        Returns:
            tensor (`tensor`): scaled tensor
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
        ind_1, px, prev_output = tensors
        px = self.ff_layer(px)
        output = self.out_units(px) + prev_output
        return output

class InvarLayer(tf.keras.layers.Layer):
    """`InvarLayer` is used for invariant features with non-linear activation. It consists of `PI-II-IP-PP` layers, which are executed sequentially."

    """
    def __init__(self, pp_nodes, pi_nodes, ii_nodes, **kwargs):
        super().__init__()
        self.pi_layer = PILayer(pi_nodes, **kwargs)
        self.ii_layer = FFLayer(ii_nodes, use_bias=False, **kwargs)
        self.ip_layer = IPLayer()
        self.pp_layer = FFLayer(pp_nodes, use_bias=False, **kwargs)

    def call(self, tensors):
        """
        InvarLayer take a list of three tensors as input:

        - ind_2: [sparse indices](layers.md#sparse-indices) of pairs with shape `(n_pairs, 2)`
        - p1: scalar tensor with shape `(n_atoms, n_prop)`
        - basis: interaction tensor with shape `(n_pairs, n_basis)`

        Args:
            tensors (list of tensors): list of `[ind_2, p1, basis]` tensors

        Returns:
            p1 (tensor): updated scalar property
            i1 (tensor): interaction tensor with shape `(n_pairs, n_nodes[-1])`
        """
        ind_2, p1, basis = tensors

        i1 = self.pi_layer([ind_2, p1, basis])
        i1 = self.ii_layer(i1)
        p1 = self.ip_layer([ind_2, p1, i1])
        p1 = self.pp_layer(p1)
        return p1, i1


class EquivarLayer(tf.keras.layers.Layer):
    """`EquivarLayer` is used for equivariant features without non-linear activation. It includes `PI-II-IP-PP` layers, along with `Scale` and `Dot` layers.

    """

    def __init__(self, n_outs, weighted=False, **kwargs):

        super().__init__()

        kw = kwargs.copy()
        kw["use_bias"] = False
        kw["activation"] = None

        self.pi_layer = PIXLayer(weighted=weighted, **kw)
        self.ii_layer = FFLayer(n_outs, **kwargs)
        self.ip_layer = IPLayer()
        self.pp_layer = FFLayer(n_outs, **kw)

        self.scale_layer = ScaleLayer()
        self.dot_layer = DotLayer(weighted=weighted)

    def call(self, tensors):
        """
        EquivarLayer take a list of four tensors as input:

        - ind_2: [sparse indices](layers.md#sparse-indices) of pairs with shape `(n_pairs, 2)`
        - px: equivariant tensor with shape `(n_atoms, n_components, n_prop)`
        - p1: scalar tensor with shape `(n_atoms, n_prop)`
        - diff: displacement vector with shape `(n_pairs, 3)`

        Args:
            tensors (list of tensors): list of `[ind_2, p1, basis]` tensors

        Returns:
            px (tensor): equivariant property with shape `(n_pairs, n_components, n_nodes[-1])`
            ix (tensor): equivariant interaction with shape `(n_pairs, n_components, n_nodes[-1])`
            dotted_px (tensor): dotted equivariant property
        """
        ind_2, px, i1, diff = tensors

        ix = self.pi_layer([ind_2, px])
        ix = self.scale_layer([ix, i1])
        scaled_diff = self.scale_layer([diff[:, :, None], i1])
        ix = ix + scaled_diff
        px = self.ip_layer([ind_2, px, ix])
        px = self.pp_layer(px)
        dotted_px = self.dot_layer(px)

        return px, ix, dotted_px


class GCBlock(tf.keras.layers.Layer):
    """This class implements the Keras Model for the PiNet2 network."""

    def __init__(self, rank, weighted: bool, pp_nodes, pi_nodes, ii_nodes, **kwargs):
        super(GCBlock, self).__init__()
        self.rank = rank
        # n_props: number of properties
        # rank 1 -> 1, rank 3 -> 2, rank 5 -> 3
        self.n_props = int(rank // 2) + 1
        ppx_nodes = [pp_nodes[-1]]
        if rank >= 1:
            ii1_nodes = ii_nodes.copy()
            pp1_nodes = pp_nodes.copy()
            ii1_nodes[-1] *= self.n_props  # for first split
            pp1_nodes[-1] = ii_nodes[-1] * self.n_props  # second split
            self.invar_p1_layer = InvarLayer(pp_nodes, pi_nodes, ii1_nodes, **kwargs)
            self.pp_layer = FFLayer(pp1_nodes, **kwargs)

        if rank >= 3:
            self.equivar_p3_layer = EquivarLayer(ppx_nodes, weighted=weighted, **kwargs)

        if rank >= 5:
            self.equivar_p5_layer = EquivarLayer(ppx_nodes, weighted=weighted, **kwargs)

        self.scale3_layer = ScaleLayer()
        self.scale5_layer = ScaleLayer()

    def call(self, tensors, basis):
        """PiNet2 takes batches atomic data as input, the following keys are
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
        ind_2 = tensors["ind_2"]

        p1, i1 = self.invar_p1_layer([ind_2, tensors["p1"], basis])

        i1s = tf.split(i1, self.n_props, axis=-1)
        px_list = [p1]
        new_tensors = {"i1": i1}

        if self.rank >= 3:
            p3, i3, dotted_p3 = self.equivar_p3_layer(
                [ind_2, tensors["p3"], i1s[1], tensors["d3"]]
            )  # NOTE: use same i1 branch for diff_px and px, same result as separated i1
            px_list.append(dotted_p3)
            new_tensors["i3"] = i3
            new_tensors["dotted_p3"] = dotted_p3

        if self.rank >= 5:
            p5, i5, dotted_p5 = self.equivar_p5_layer(
                [ind_2, tensors["p5"], i1s[2], tensors["d5"]]
            )
            px_list.append(dotted_p5)
            new_tensors["i5"] = i5
            new_tensors["dotted_p5"] = dotted_p5

        p1t1 = self.pp_layer(  # 1P''(i3r) -> fflayer -> 1P'(i3r)
            tf.concat(  # 1P''(i3r) := [1P' 3P' 5P']
                px_list,
                axis=-1,
            )
        )

        pxt1 = tf.split(p1t1, self.n_props, axis=-1)
        new_tensors["p1"] = pxt1[0]
        if self.rank >= 3:
            p3t1 = self.scale3_layer([p3, pxt1[1]])
            new_tensors["p3"] = p3t1

        if self.rank >= 5:
            p5t1 = self.scale5_layer([p5, pxt1[2]])
            new_tensors["p5"] = p5t1

        return new_tensors


class PreprocessLayer(tf.keras.layers.Layer):
    def __init__(self, rank, atom_types, rc):
        super(PreprocessLayer, self).__init__()
        self.rank = rank
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
        out_extra={},
        out_pool=False,
        act="tanh",
        depth=4,
        weighted=False,
        rank=3,
    ):
        """
        Args:
            atom_types (list): elements for the one-hot embedding
            rc (float): cutoff radius
            cutoff_type (string): cutoff function to use with the basis.
            basis_type (string): basis function, can be "polynomial" or "gaussian"
            n_basis (int): number of basis functions to use
            gamma (float or array): width of gaussian function for gaussian basis
            center (float or array): center of gaussian function for gaussian basis
            pp_nodes (list): number of nodes for PPLayer
            pi_nodes (list): number of nodes for PILayer
            ii_nodes (list): number of nodes for IILayer
            out_nodes (list): number of nodes for OutLayer
            out_units (int): number of output feature
            out_extra (dict[str, int]): return extra variables
            out_pool (str): pool atomic outputs, see ANNOutput
            act (string): activation function to use
            depth (int): number of interaction blocks
            weighted (bool): whether to use weighted style
            rank (int[1, 3, 5]): which order of variable to use
        """
        super(PiNet2, self).__init__()

        self.depth = depth
        assert rank in [1, 3, 5], ValueError("rank must be 1, 3, or 5")
        self.rank = rank
        self.preprocess = PreprocessLayer(rank, atom_types, rc)
        self.cutoff = CutoffFunc(rc, cutoff_type)

        if basis_type == "polynomial":
            self.basis_fn = PolynomialBasis(n_basis)
        elif basis_type == "gaussian":
            self.basis_fn = GaussianBasis(center, gamma, rc, n_basis)

        if rank >= 1:
            self.res_update1 = [ResUpdate() for _ in range(depth)]
        if rank >= 3:
            self.res_update3 = [ResUpdate() for _ in range(depth)]
        if rank >= 5:
            self.res_update5 = [ResUpdate() for _ in range(depth)]
        self.gc_blocks = [
            GCBlock(rank, weighted, pp_nodes, pi_nodes, ii_nodes, activation=act)
            for _ in range(depth)
        ]
        self.out_layers = [OutLayer(out_nodes, out_units) for i in range(depth)]
        self.out_extra = out_extra
        for k, v in out_extra.items():
            setattr(
                self, f"{k}_out_layers", [OutLayer(out_nodes, v) for i in range(depth)]
            )
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
        ind_1 = tensors["ind_1"]
        tensors["d3"] = tensors["diff"] / tf.linalg.norm(tensors["diff"], axis=-1, keepdims=True)
        if self.rank >= 3:
            tensors["p3"] = tf.zeros([tf.shape(ind_1)[0], 3, 1])
        if self.rank >= 5:
            tensors["p5"] = tf.zeros([tf.shape(ind_1)[0], 5, 1])

            diff = tensors["d3"]
            x = diff[:, 0]
            y = diff[:, 1]
            z = diff[:, 2]
            x2 = x**2
            y2 = y**2
            z2 = z**2
            tensors["d5"] = tf.stack(
                [
                    2 / 3 * x2 - 1 / 3 * y2 - 1 / 3 * z2,
                    2 / 3 * y2 - 1 / 3 * x2 - 1 / 3 * z2,
                    x * y,
                    x * z,
                    y * z,
                ],
                axis=1,
            )

        fc = self.cutoff(tensors["dist"])
        basis = self.basis_fn(tensors["dist"], fc=fc)
        output = 0.0
        out_extra = {k: 0.0 for k in self.out_extra}
        for i in range(self.depth):
            new_tensors = self.gc_blocks[i](tensors, basis)
            output = self.out_layers[i]([ind_1, new_tensors["p1"], output])
            for k in self.out_extra:
                out_extra[k] = getattr(self, f"{k}_out_layers")[i](
                    [ind_1, new_tensors[k], out_extra[k]]
                )
            if self.rank >= 1:
                tensors["p1"] = self.res_update1[i]([tensors["p1"], new_tensors["p1"]])
            if self.rank >= 3:
                tensors["p3"] = self.res_update3[i]([tensors["p3"], new_tensors["p3"]])
            if self.rank >= 5:
                tensors["p5"] = self.res_update5[i]([tensors["p5"], new_tensors["p5"]])

        output = self.ann_output([ind_1, output])
        if self.out_extra:
            return output, out_extra
        else:
            return output

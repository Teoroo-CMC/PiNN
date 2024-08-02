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
from pinn.networks.pinet2 import PIXLayer, ScaleLayer, OutLayer, DotLayer


class InvarLayer(tf.keras.layers.Layer):

    def __init__(self, pp_nodes, pi_nodes, ii_nodes, **kwargs):
        super().__init__()
        self.pi_layer = PILayer(pi_nodes, **kwargs)
        self.ii_layer = FFLayer(ii_nodes, use_bias=False, **kwargs)
        self.ip_layer = IPLayer()
        self.pp_layer = FFLayer(pp_nodes, use_bias=False, **kwargs)

    def call(self, tensors):

        ind_2, p1, basis = tensors

        i1 = self.pi_layer([ind_2, p1, basis])
        i1 = self.ii_layer(i1)
        p1 = self.ip_layer([ind_2, p1, i1])
        p1 = self.pp_layer(p1)
        return p1, i1


class EquiVarLayer(tf.keras.layers.Layer):

    def __init__(self, n_outs, weighted=False, **kwargs):

        super().__init__()

        kw = kwargs.copy()
        kw["use_bias"] = False
        kw["activation"] = None

        self.pp_layer = FFLayer(n_outs, **kw)
        self.pi_layer = PIXLayer(weighted=weighted, **kw)
        self.ip_layer = IPLayer()

        self.scale_layer = ScaleLayer()
        self.dot_layer = DotLayer(weighted=weighted)

    def call(self, tensors):

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
    def __init__(self, rank, weighted: bool, pp_nodes, pi_nodes, ii_nodes, **kwargs):
        super(GCBlock, self).__init__()
        self.rank = rank
        self.n_props = int(rank // 2) + 1
        ppx_nodes = [pp_nodes[-1]]
        if rank >= 1:
            ii1_nodes = ii_nodes.copy()
            pp1_nodes = pp_nodes.copy()
            ii1_nodes[-1] *= self.n_props
            pp1_nodes[-1] = ii_nodes[-1] * self.n_props
            self.invar_p1_layer = InvarLayer(pp_nodes, pi_nodes, ii1_nodes, **kwargs)
            self.pp_layer = FFLayer(pp1_nodes, **kwargs)

        if rank >= 3:
            self.equivar_p3_layer = EquiVarLayer(ppx_nodes, weighted=weighted, **kwargs)

        if rank >= 5:
            self.equivar_p5_layer = EquiVarLayer(ppx_nodes, weighted=weighted, **kwargs)

        self.scale3_layer = ScaleLayer()
        self.scale5_layer = ScaleLayer()

    def call(self, tensors, basis):

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

        p1t1 = self.pp_layer(
            tf.concat(
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
        weighted=True,
        rank=3,
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
            out_extra (dict[str, int]): return extra variables.
            cutoff_type (string): cutoff function to use with the basis.
            act (string): activation function to use
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

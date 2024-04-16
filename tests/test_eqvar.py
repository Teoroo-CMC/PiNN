from pinn.networks.pinet import PiNet
from pinn.networks.pinet2 import PiNet2
from pinn.networks.pinet2_p5_dot import PiNet2P5Dot
from pinn.networks.pinet2_p5_prod import PiNet2P5Prod
from pinn.networks.pinet2_p5_irrep_dot import PiNet2P5IrrepDot
from pinn.networks.pinet2_p5_irrep_prod import PiNet2P5IrrepProd
# from pinn.networks.pinet2_p5_irrep_combine import PiNet2P5IrrepCombine
import pytest
import tensorflow as tf
import numpy as np
import random

def create_rot_mat(theta):
    return tf.constant([
        [1., 0., 0.],
        [0., np.cos(theta), -np.sin(theta)],
        [0., np.sin(theta), np.cos(theta)]
    ], dtype=tf.float32)

def rotate(x, theta):
    rot = create_rot_mat(theta)
    return tf.einsum('ix,xy->iy', x, rot)


class TestEquivar:

    def test_pinet(self, mocked_data):
        pinet = PiNet(
            atom_types=[0, 1],
        )

        for batch in mocked_data:
            batch1 = batch.copy()
            energy1 = pinet(batch1)

            batch2 = batch.copy()
            batch['coord'] = rotate(batch2['coord'], 42.)
            energy2 = pinet(batch)
            tf.debugging.assert_near(energy1, energy2)
        
    def test_pinet2(self, mocked_data):
        pinet = PiNet2(
            atom_types=[0, 1],
        )

        for batch in mocked_data:
            batch1 = batch.copy()
            energy1 = pinet(batch1)

            batch2 = batch.copy()
            batch2['coord'] = rotate(batch2['coord'], 42.)
            energy2 = pinet(batch2)
            tf.debugging.assert_near(energy1, energy2)

    def test_pinet2_p5_dot(self, mocked_data):
        pinet = PiNet2P5Dot(
            atom_types=[0, 1],
        )

        for batch in mocked_data:
            batch1 = batch.copy()
            energy1 = pinet(batch1)

            batch2 = batch.copy()
            batch2['coord'] = rotate(batch2['coord'], 42.)
            energy2 = pinet(batch2)
            np.testing.assert_allclose(energy1, energy2, rtol=1e-4, atol=1e-4)

    def test_pinet2_p5_prod(self, mocked_data):
        pinet = PiNet2P5Prod(
            atom_types=[0, 1],
        )

        for batch in mocked_data:
            batch1 = batch.copy()
            energy1 = pinet(batch1)

            batch2 = batch.copy()
            batch2['coord'] = rotate(batch2['coord'], 42.)
            energy2 = pinet(batch2)
            np.testing.assert_allclose(energy1, energy2, rtol=1e-3, atol=1e-3)

    def test_pinet2_p5_irrep_dot(self, mocked_data):
        pinet = PiNet2P5IrrepDot(
            atom_types=[0, 1],
        )
        import random
        for batch in mocked_data:
            batch1 = batch.copy()
            energy1 = pinet(batch1)

            batch2 = batch.copy()
            batch2['coord'] = rotate(tf.cast(batch2['coord'], tf.float32), random.randint(0, 4123))
            energy2 = pinet(batch2)
            np.testing.assert_allclose(energy1, energy2, rtol=1e-4, atol=1e-4)

    def test_pinet2_p5_irrep_prod(self, mocked_data):
        pinet = PiNet2P5IrrepProd(
            atom_types=[0, 1],
        )
        import random
        for batch in mocked_data:
            batch1 = batch.copy()
            energy1 = pinet(batch1)

            batch2 = batch.copy()
            batch2['coord'] = rotate(tf.cast(batch2['coord'], tf.float32), random.randint(0, 4123))
            energy2 = pinet(batch2)
            np.testing.assert_allclose(energy1, energy2, rtol=1e-4, atol=1e-4)


    def test_pinet2_p5_irrep_prod_internal(self, mocked_data):
        from pinn.layers import (
            CellListNL,
            CutoffFunc,
            PolynomialBasis,
            GaussianBasis,
            AtomicOnehot,
            ANNOutput,
        )
        from pinn.networks.pinet import ResUpdate
        from pinn.networks.pinet2_p5_dot import OutLayer
        from pinn.networks.pinet2_p5_irrep_prod import GCBlock, PreprocessLayer

        atom_types = [0, 1]
        depth = 5
        rc = 5.0
        cutoff_type = 'f1'
        basis_type = "polynomial"
        n_basis = 4
        gamma = 3.0
        center = None

        pp_nodes=[16, 16]
        pi_nodes=[16, 16]
        ii_nodes=[16, 16]
        out_nodes=[16, 16]
        out_units=1
        out_pool=False
        act="tanh"
        weighted=True

        preprocess = PreprocessLayer(atom_types, rc)
        cutoff = CutoffFunc(rc, cutoff_type)

        if basis_type == "polynomial":
            basis_fn = PolynomialBasis(n_basis)
        elif basis_type == "gaussian":
            basis_fn = GaussianBasis(center, gamma, rc, n_basis)

        res_update1 = [ResUpdate() for i in range(depth)]
        res_update3 = [ResUpdate() for i in range(depth)]
        res_update5 = [ResUpdate() for i in range(depth)]
        gc_blocks = [GCBlock(weighted, [], pi_nodes, ii_nodes, activation=act)]
        gc_blocks += [
            GCBlock(weighted, pp_nodes, pi_nodes, ii_nodes, activation=act)
            for i in range(depth - 1)
        ]
        out_layers = [OutLayer(out_nodes, out_units) for i in range(depth)]
        # ann_output = ANNOutput(out_pool)

        def call(tensors):
            tensors = preprocess(tensors)
            tensors["p3"] = tf.zeros([tf.shape(tensors["ind_1"])[0], 3, 1])
            tensors["p5"] = tf.zeros([tf.shape(tensors["ind_1"])[0], 5, 1])
            fc = cutoff(tensors["dist"])
            basis = basis_fn(tensors["dist"], fc=fc)
            output = 0.0
            for i in range(depth):
                p1, p3, p5 = gc_blocks[i](
                    [tensors["ind_2"], tensors["p1"], tensors["p3"], tensors["p5"], tensors["norm_diff"], tensors["diff_p5"], basis]
                )
                output = out_layers[i]([tensors["ind_1"], p1, p3, output])
                tensors["p1"] = res_update1[i]([tensors["p1"], p1])
                tensors["p3"] = res_update3[i]([tensors["p3"], p3])
                tensors["p5"] = res_update5[i]([tensors["p5"], p5])

            return tensors

        def rotate_batch(x, theta):
            ndim = x.ndim - 2
            rot = create_rot_mat(theta)
            if ndim == 0:
                return tf.einsum('xc, xy->yc', x, rot)
            elif ndim == 1:
                return tf.einsum('ixc, xy->iyc', x, rot)
            elif ndim == 2:
                return tf.einsum('xw, ixyc, yz->iwzc', rot, x, rot)

        for batch in mocked_data:
            rnd = random.randint(0, 42)
    
            batch1 = batch.copy()
            tensors1 = call(batch1)

            batch2 = batch.copy()
            batch2['coord'] = rotate(tf.cast(batch2['coord'], tf.float32), rnd)
            tensors2 = call(batch2)

            np.testing.assert_allclose(tensors1['p1'], tensors2['p1'], rtol=1e-4, atol=1e-4)
            np.testing.assert_allclose(
                rotate_batch(tensors1['p3'], rnd), tensors2['p3'], rtol=1e-4, atol=1e-4
            )



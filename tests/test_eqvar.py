from pinn.networks.pinet import PiNet
from pinn.networks.pinet2 import PiNet2
from pinn.networks.pinet2_p5_dot import PiNet2P5Dot
from pinn.networks.pinet2_p5_prod import PiNet2P5Prod
from pinn.networks.pinet2_p5_irrep_dot import PiNet2P5IrrepDot
from pinn.networks.pinet2_p5_irrep_prod import PiNet2P5IrrepProd
from pinn.networks.pinet2_p5_irrep_combine import PiNet2P5IrrepCombine
import pytest
import tensorflow as tf
import numpy as np

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
            print(batch2['coord'].shape)
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
            np.testing.assert_allclose(energy1, energy2, rtol=1e-2, atol=1e-3)


import tensorflow as tf
import numpy as np
import pytest
from pinn.networks.pinet import PiNet
from pinn.networks.pinet2 import PiNet2
from utils import rotate


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

    @pytest.mark.parametrize('rank', [1, 3, 5])
    def test_pinet2_refactor(self, mocked_data, rank):
        from pinn.networks.pinet2 import PiNet2

        pinet = PiNet2(
            rank=rank,
            atom_types=[0, 1],
        )

        for batch in mocked_data:
            batch1 = batch.copy()
            energy1 = pinet(batch1)

            batch2 = batch.copy()
            batch2['coord'] = rotate(batch2['coord'], 42.)
            energy2 = pinet(batch2)
            tf.debugging.assert_near(energy1, energy2)

    def test_pinet2_refactor(self, mocked_data):
        from pinn.networks.pinet2 import PiNet2

        pinet = PiNet2(
            rank=3,
            atom_types=[0, 1],
            out_extra={
                f'p3': 16,
            }
        )

        for batch in mocked_data:
            batch1 = batch.copy()
            energy1, actual_extra = pinet(batch1)

            batch2 = batch.copy()
            batch2['coord'] = rotate(batch2['coord'], 42.)
            energy2, expect_extra = pinet(batch2)
            tf.debugging.assert_near(energy1, energy2)
            tf.debugging.assert_near(rotate(actual_extra['p3'], 42), expect_extra['p3'])
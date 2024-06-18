import pytest
import tensorflow as tf
import numpy as np
import numpy.testing as npt
import random
from pinn.networks.pinet2 import DotLayer, ScaleLayer, PIXLayer
from pinn.networks.pinet2_modularized import EquiVarLayer
from utils import rotate


class TestPiNet2:

    @pytest.mark.forked
    def test_pixlayer_rank2(self):

        n_atoms = 10
        n_dims = 3
        n_channels = 5
        n_pairs = 3
        px = tf.random.uniform((n_atoms, n_dims, n_dims, n_channels))
        ind_2 = tf.random.uniform((n_pairs, 2), maxval=n_atoms, dtype=tf.int32)

        layer = PIXLayer(weighted=True)
        theta = 42
        actual = layer([ind_2, rotate(px, theta)])
        expect = rotate(layer([ind_2, px]), theta)
        tf.debugging.assert_near(actual, expect)

    @pytest.mark.forked
    def test_dotlayer_rank2(self):

        n_atoms = 10
        n_dims = 3
        n_channels = 5

        prop = tf.random.uniform((n_atoms, n_dims, n_dims, n_channels))
        prop_shape = prop.shape
        dot = DotLayer(weighted=True)
        actual = dot(
            tf.reshape(rotate(prop, 42.0), (prop_shape[0], -1, prop_shape[-1]))
        )
        expect = dot(tf.reshape(prop, (prop_shape[0], -1, prop_shape[-1])))
        tf.debugging.assert_near(actual, expect)

    @pytest.mark.forked
    def test_scalelayer_rank2(self):

        n_atoms = 10
        n_dims = 3
        n_channels = 5

        px = tf.random.uniform((n_atoms, n_dims, n_dims, n_channels))
        p1 = tf.random.uniform((n_atoms, n_channels))

        scaler = ScaleLayer()
        out = scaler([px, p1[:, None, :]])
        assert out.shape == (n_atoms, n_dims, n_dims, n_channels)
        tf.debugging.assert_near(
            rotate(scaler([px, p1[:, None, :]]), 42.0),
            scaler([rotate(px, 42.0), p1[:, None, :]]),
        )

    @pytest.mark.forked
    def test_simple_dotlayer(self):

        n_atoms = 10
        n_dims = 3
        n_channels = 5

        prop = tf.random.uniform((n_atoms, n_dims, n_channels))

        dot = DotLayer(weighted=False)
        actual = dot(rotate(prop, 42.0))
        expect = dot(prop)
        tf.debugging.assert_near(actual, expect)

    @pytest.mark.forked
    def test_general_dotlayer(self):

        n_atoms = 10
        n_dims = 3
        n_channels = 5

        prop = tf.random.uniform((n_atoms, n_dims, n_channels))
        theta = 42.0

        dot = DotLayer(weighted=True)
        tf.debugging.assert_near(dot(prop), dot(rotate(prop, theta)))

    @pytest.mark.forked
    def test_scalelayer(self):

        n_atoms = 10
        n_dims = 3
        n_channels = 5

        px = tf.random.uniform((n_atoms, n_dims, n_channels))
        p1 = tf.random.uniform((n_atoms, n_channels))

        scaler = ScaleLayer()
        out = scaler([px, p1])
        assert out.shape == (n_atoms, n_dims, n_channels)
        tf.debugging.assert_near(
            rotate(scaler([px, p1]), 42.), scaler([rotate(px, 42.), p1])
        )

    @pytest.mark.forked
    def test_simple_pixlayer_rank1(self):

        n_atoms = 10
        n_dims = 3
        n_channels = 5
        n_pairs = 3
        px = tf.random.uniform((n_atoms, n_dims, n_channels))
        ind_2 = tf.random.uniform((n_pairs, 2), maxval=n_atoms, dtype=tf.int32)

        pix = PIXLayer(weighted=False)
        out = pix([ind_2, px])
        assert out.shape == (n_pairs, n_dims, n_channels)
        tf.debugging.assert_near(
            rotate(pix([ind_2, px]), 42.0), pix([ind_2, rotate(px, 42.0)])
        )

    @pytest.mark.forked
    def test_general_pixlayer(self):

        n_atoms = 10
        n_dims = 3
        n_channels = 5
        n_pairs = 3
        px = tf.random.uniform((n_atoms, n_dims, n_channels))
        ind_2 = tf.random.uniform((n_pairs, 2), maxval=n_atoms, dtype=tf.int32)

        pix = PIXLayer(weighted=True)
        out = pix([ind_2, px])
        assert out.shape == (n_pairs, n_dims, n_channels)
        tf.debugging.assert_near(
            rotate(pix([ind_2, px]), 42.0), pix([ind_2, rotate(px, 42.0)])
        )
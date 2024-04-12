from math import exp
from pinn.networks.pinet2 import DotLayer, ScaleLayer, PIXLayer
from pinn.networks.pinet import PILayer
import pytest
import tensorflow as tf
import numpy as np

from pinn.networks.pinet2_p5_prod import AddLayer, TensorProductLayer


def create_rot_mat(theta):
    return tf.constant([
        [1., 0., 0.],
        [0., np.cos(theta), -np.sin(theta)],
        [0., np.sin(theta), np.cos(theta)]
    ], dtype=tf.float32)

def rotate(x, theta):
    ndim = x.ndim - 2
    rot = create_rot_mat(theta)
    if ndim == 0:
        return tf.einsum('xc, xy->yc', x, rot)
    elif ndim == 1:
        return tf.einsum('ixc, xy->iyc', x, rot)
    elif ndim == 2:
        return tf.einsum('xw, ixyc, yz->iwzc', rot, x, rot)

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
        tf.debugging.assert_near(
            actual, expect
        )

    @pytest.mark.forked
    def test_dotlayer_rank2(self):

        n_atoms = 10
        n_dims = 3
        n_channels = 5

        prop = tf.random.uniform((n_atoms, n_dims, n_dims, n_channels))
        prop_shape = prop.shape
        dot = DotLayer(weighted=True)
        actual = dot(tf.reshape(rotate(prop, 42.), (prop_shape[0], -1, prop_shape[-1])))
        expect = dot(tf.reshape(prop, (prop_shape[0], -1, prop_shape[-1])))
        tf.debugging.assert_near(
            actual, expect
        )        

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
            rotate(scaler([px, p1[:, None, :]]), 42.), scaler([rotate(px, 42.), p1[:, None, :]])
        )

    @pytest.mark.forked
    def test_addlayer_rank2(self):

        n_atoms = 10
        n_dims = 3
        n_channels = 5
        n_pairs = 12

        px = tf.random.uniform((n_atoms, n_dims, n_dims, n_channels))
        i3 = tf.random.uniform((n_pairs, 3, n_channels))
        ind_2 = tf.random.uniform((n_pairs, 2), maxval=n_atoms, dtype=tf.int32)
        ind_i = ind_2[:, 0]
        ind_j = ind_2[:, 1]
        # ix_i = tf.gather(px, ind_i)
        ix_j = tf.gather(px, ind_j)

        layer = AddLayer()
        out = layer([ix_j, i3])
        assert out.shape == (n_pairs, n_dims, n_dims, n_channels)
        np.testing.assert_allclose(
            rotate(layer([ix_j, i3]), 42.), layer([rotate(ix_j, 42.), rotate(i3, 42.)]), rtol=1e-5, atol=1e-5
        )

    @pytest.mark.forked
    def test_prodlayer_rank2(self):

        n_atoms = 10
        n_dims = 3
        n_channels = 5

        p3 = tf.random.uniform((n_atoms, n_dims, n_channels))
        p5 = tf.random.uniform((n_atoms, n_dims, n_dims, n_channels))

        layer = TensorProductLayer()
        out = layer([p3, p5])
        assert out.shape == (n_atoms, n_channels)
        actual = layer([p3, p5])
        expect = layer([rotate(p3, 42.), rotate(p5, 42.)])
        np.testing.assert_allclose(
            actual, expect, rtol=1e-5, atol=1e-5
        )   

    @pytest.mark.forked
    def test_simple_dotlayer(self):

        n_atoms = 10
        n_dims = 3
        n_channels = 5

        prop = tf.random.uniform((n_atoms, n_dims, n_channels))

        dot = DotLayer('simple')
        actual = dot(rotate(prop, 42.))
        expect = dot(prop)
        tf.debugging.assert_near(
            actual, expect
        )

    @pytest.mark.forked
    def test_general_dotlayer(self):

        n_atoms = 10
        n_dims = 3
        n_channels = 5

        prop = tf.random.uniform((n_atoms, n_dims, n_channels))
        theta = 42.

        dot = DotLayer('general')
        tf.debugging.assert_near(
            dot(prop), dot(rotate(prop, theta))
        )

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

        pix = PIXLayer('simple')
        out = pix([ind_2, px])
        assert out.shape == (n_pairs, n_dims, n_channels)
        tf.debugging.assert_near(
            rotate(pix([ind_2, px]), 42.), pix([ind_2, rotate(px, 42.)])
        )

    @pytest.mark.forked
    def test_general_pixlayer(self):

        n_atoms = 10
        n_dims = 3
        n_channels = 5
        n_pairs = 3
        px = tf.random.uniform((n_atoms, n_dims, n_channels))
        ind_2 = tf.random.uniform((n_pairs, 2), maxval=n_atoms, dtype=tf.int32)

        pix = PIXLayer('general')
        out = pix([ind_2, px])
        assert out.shape == (n_pairs, n_dims, n_channels)
        tf.debugging.assert_near(
            rotate(pix([ind_2, px]), 42.), pix([ind_2, rotate(px, 42.)])
        )

 
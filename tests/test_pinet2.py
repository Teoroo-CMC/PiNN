from pinn.networks.pinet2 import DotLayer, ScaleLayer, PIXLayer
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
    return tf.einsum('ixa,xy->iya', x, rot)

class TestPiNet2:

    @pytest.mark.forked
    def test_simple_dotlayer(self):

        nsamples = 10
        ndims = 3
        nchannels = 5

        prop = tf.random.uniform((nsamples, ndims, nchannels))

        dot = DotLayer(weighted=False)
        tf.debugging.assert_near(
            dot(prop), dot(rotate(prop, 42.))
        )

    @pytest.mark.forked
    def test_general_dotlayer(self):

        nsamples = 10
        ndims = 3
        nchannels = 5

        prop = tf.random.uniform((nsamples, ndims, nchannels))
        theta = 42.

        dot = DotLayer(weighted=True)
        tf.debugging.assert_near(
            dot(prop), dot(rotate(prop, theta))
        )

    @pytest.mark.forked
    def test_scalelayer(self):

        nsamples = 10
        ndims = 3
        nchannels = 5

        px = tf.random.uniform((nsamples, ndims, nchannels))
        p1 = tf.random.uniform((nsamples, nchannels))

        scaler = ScaleLayer()
        out = scaler([px, p1])
        assert out.shape == (nsamples, ndims, nchannels)
        tf.debugging.assert_near(
            rotate(scaler([px, p1]), 42.), scaler([rotate(px, 42.), p1])
        )

    @pytest.mark.forked
    def test_simple_pixlayer(self):

        nsamples = 10
        ndims = 3
        nchannels = 5
        nnbors = 3
        px = tf.random.uniform((nsamples, ndims, nchannels))
        ind_2 = tf.random.uniform((nnbors, 2), maxval=nsamples, dtype=tf.int32)

        pix = PIXLayer(weighted=False)
        out = pix([ind_2, px])
        assert out.shape == (nnbors, ndims, nchannels)
        tf.debugging.assert_near(
            rotate(pix([ind_2, px]), 42.), pix([ind_2, rotate(px, 42.)])
        )


    @pytest.mark.forked
    def test_general_pixlayer(self):

        nsamples = 10
        ndims = 3
        nchannels = 5
        nnbors = 3
        px = tf.random.uniform((nsamples, ndims, nchannels))
        ind_2 = tf.random.uniform((nnbors, 2), maxval=nsamples, dtype=tf.int32)

        pix = PIXLayer(weighted=True)
        out = pix([ind_2, px])
        assert out.shape == (nnbors, ndims, nchannels)
        tf.debugging.assert_near(
            rotate(pix([ind_2, px]), 42.), pix([ind_2, rotate(px, 42.)])
        )

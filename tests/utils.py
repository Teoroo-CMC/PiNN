import tensorflow as tf
import numpy as np


def create_rot_mat(theta):
    return tf.constant(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(theta), -np.sin(theta)],
            [0.0, np.sin(theta), np.cos(theta)],
        ],
        dtype=tf.float32,
    )


def rotate(x, theta):
    ndim = x.ndim - 2
    rot = create_rot_mat(theta)
    if ndim == 0:
        return tf.einsum('ix,xy->iy', x, rot)
    elif ndim == 1:
        return tf.einsum("ixc, xy->iyc", x, rot)
    elif ndim == 2:
        return tf.einsum("xw, ixyc, yz->iwzc", rot, x, rot)

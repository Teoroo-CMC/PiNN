# -*- coding: utf-8 -*-

"""(Keras) Layers for radial basis functions"""

import numpy as np
import tensorflow as tf


class CutoffFunc(tf.keras.layers.Layer):
    """returns the cutoff function of given type

    Args:
        dist (tensor): a tensor of distance
        cutoff_type (string): name of the cutoff function
        rc (float): cutoff radius

    Returns:
        A cutoff function tensor with the same shape of dist
    """

    def __init__(self, rc=5.0, cutoff_type="f1"):
        super(CutoffFunc, self).__init__()
        self.cutoff_type = cutoff_type
        self.rc = rc
        f1 = lambda x: 0.5 * (tf.cos(np.pi * x / rc) + 1)
        f2 = lambda x: (tf.tanh(1 - x / rc) / np.tanh(1)) ** 3
        hip = lambda x: tf.cos(np.pi * x / rc / 2) ** 2
        self.cutoff_fn = {"f1": f1, "f2": f2, "hip": hip}[cutoff_type]

    def call(self, distance):
        return self.cutoff_fn(distance)


class GaussianBasis(tf.keras.layers.Layer):
    """Gaussian Basis Layer

    Transforms distances to a set of gaussian basis
    """

    def __init__(self, center=None, gamma=None, rc=None, n_basis=None):
        """
        Initialize the Gaussian basis layer.

        n_basis and rc are only used when center is not given

        Args:
            center (list/array of float): Gaussian centers
            gamma (list/array of float): inverse Gaussian width
            rc: cutoff radius
            n_basis: number of basis function
        """
        super(GaussianBasis, self).__init__()
        if center is None:
            self.center = np.linspace(0, rc, n_basis)
        else:
            self.center = np.array(center)
        self.gamma = np.broadcast_to(gamma, self.center.shape)

    def call(self, dist, fc=None):
        basis = tf.stack(
            [
                tf.exp(-gamma * (dist - center) ** 2)
                for (center, gamma) in zip(self.center, self.gamma)
            ],
            axis=1,
        )
        if fc is not None:
            basis = tf.einsum("pb,p->pb", basis, fc)  # p-> pair; b-> basis
        return basis


class PolynomialBasis(tf.keras.layers.Layer):
    """Polynomial Basis Layer

    Transforms distances to a set of polynomial basis
    """

    def __init__(self, n_basis):
        """
        Initialize the Polynomial Basis

        n_basis can be a list of explicitly specified polynomail orders

        Args:
            n_basis: number of basis function
        """
        super(PolynomialBasis, self).__init__()
        if type(n_basis) != list:
            n_basis = [(i + 1) for i in range(n_basis)]
        self.n_basis = n_basis

    def call(self, dist, fc=None):
        assert fc is not None, "Polynomail basis requires a cutoff function."
        basis = tf.stack([fc**i for i in self.n_basis], axis=1)
        return basis

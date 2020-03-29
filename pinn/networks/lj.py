# -*- coding: utf-8 -*-
import tensorflow as tf
from pinn.layers import CellListNL


class LJ(tf.keras.Model):
    """Lennard-Jones Potential

    This is a simple implementation of LJ potential
    for the purpose of testing distance/force calculations.

    Args:
        tensors: input data (nested tensor from dataset).
        rc: cutoff radius.
        sigma, epsilon: LJ parameters
    """
    def __init__(self, rc=3.0, sigma=1.0, epsilon=1.0):
        super(LJ, self).__init__()
        self.rc = rc
        self.sigma = sigma
        self.epsilon = epsilon
        self.nl_layer = CellListNL(rc)

    def preprocess(self, tensors):
        if 'ind_2' not in tensors:
            tensors.update(self.nl_layer(tensors))
        return tensors

    def call(self, tensors):
        rc, sigma, epsilon = self.rc, self.sigma, self.epsilon
        tensors = self.preprocess(tensors)
        e0 = 4 * epsilon * ((sigma / rc)**12 - (sigma / rc)**6)
        c6 = (sigma/tensors['dist'])**6
        c12 = c6 ** 2
        en = 4*epsilon*(c12-c6)-e0
        natom = tf.shape(tensors['ind_1'])[0]
        nbatch = tf.reduce_max(tensors['ind_1'])+1
        en = tf.math.unsorted_segment_sum(en, tensors['ind_2'][:, 0], natom)
        return en/2.0

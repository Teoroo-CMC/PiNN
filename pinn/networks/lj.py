# -*- coding: utf-8 -*-
import tensorflow as tf
from pinn.layers import cell_list_nl
from pinn.utils import connect_dist_grad


def lj(tensors, rc=3.0, sigma=1.0, epsilon=1.0):
    """Lennard-Jones Potential

    This is a simple implementation of LJ potential
    for the purpose of testing distance/force calculations
    rather than a network.

    Args:
        tensors: input data (nested tensor from dataset).
        rc: cutoff radius.
        sigma, epsilon: LJ parameters
    """
    tensors.update(cell_list_nl(tensors, rc=rc))
    connect_dist_grad(tensors)
    e0 = 4 * epsilon * ((sigma / rc)**12 - (sigma / rc)**6)
    c6 = (sigma/tensors['dist'])**6
    c12 = c6 ** 2
    en = 4*epsilon*(c12-c6)-e0
    natom = tf.shape(tensors['ind_1'])[0]
    nbatch = tf.reduce_max(tensors['ind_1'])+1
    en = tf.unsorted_segment_sum(en, tensors['ind_2'][:, 0], natom)
    return en/2.0

"""
    Filters are layers without trainable variables
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import tensorflow as tf
import numpy as np


class atomic_mask():
    """Atomic filter

    Boolean for existing atoms
    """

    def parse(self, tensors, dtype):
        a_mask = tf.cast(tensors['atoms'], tf.bool)
        tensors['a_mask'] = a_mask


class atomic_dress():
    """Atomic dress

    Assign an energy for each type of atom
    """

    def __init__(self, dress):
        self.dress = dress

    def parse(self, tensors, dtype):
        atoms = tensors['atoms']
        energy = tf.reduce_sum([
            tf.where(tf.equal(atoms, key),
                     val * tf.cast(tf.ones_like(atoms), dtype),
                     tf.cast(tf.zeros_like(atoms), dtype))
            for key, val in self.dress.items()], [-1, -2])
        tensors['energy'] = energy


class distance_mat():
    """Distance filter
    Generates a distance tensor from the coordinates
    """

    def parse(self, tensors, dtype):
        coord = tensors['coord']
        diff = tf.expand_dims(coord, -2) - tf.expand_dims(coord, -3)

        if 'cell' in tensors and tensors['cell'].shape == [3]:
            diff = diff - tf.rint(diff / tensors['cell']) * tensors['cell']
        # elif tensors['cell'].shape == [3, 3]:
        #TODO Implement PBC for triclinic cells

        square = tf.reduce_sum(tf.square(diff), axis=-1)
        # To make the distance differentiable
        zeros = tf.equal(square, 0)
        square = tf.where(zeros,  tf.ones_like(square), square)
        dist = tf.where(zeros, tf.sqrt(square), tf.zeros_like(square))
        tensors['dist'] = square


class pi_atomic():
    """

    """

    def __init__(self, types):
        self.types = types

    def parse(self, tensors, dtype):
        shape = [1] * (len(tensors['atoms'].shape))
        shape.append(len(self.types))
        types = tf.reshape(self.types, shape)
        atoms = tf.expand_dims(tensors['atoms'], -1)
        p_nodes = tf.cast(tf.equal(types, atoms),
                          dtype)

        tensors['nodes'] = {}
        tensors['nodes'][0] = p_nodes


class pi_kernel():
    """

    """

    def __init__(self, func='f1', order=2, rc=4.0):
        self.rc = rc
        self.func = func
        self.order = order

    def parse(self, tensors, dtype):
        dist = tensors['dist']
        symm_func = tf.expand_dims(dist, -1)
        symm_func = {
            'f1': lambda x: 0.5*(tf.cos(np.pi*symm_func/self.rc)+1),
            'f2': lambda x: tf.tanh(1-symm_func/self.rc) ** 3
        }[self.func](symm_func)

        kernel = tf.concat([symm_func**(i+1)
                            for i in range(self.order)], axis=-1)

        kernel = tf.expand_dims(kernel, -2)
        p_mask = tf.expand_dims(tf.cast(tensors['a_mask'], dtype), -1)
        i_mask = tf.expand_dims(
            tf.cast((dist > 0) & (dist < self.rc), dtype), -1)

        tensors['pi_kernel'] = {1: kernel}
        tensors['pi_masks'] = {0: p_mask, 1: i_mask}

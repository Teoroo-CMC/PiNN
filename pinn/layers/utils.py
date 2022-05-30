#!/usr/bin/env python3

import tensorflow as tf

class AtomicOnehot(tf.keras.layers.Layer):
    """ One-hot encoding Lyaer

    perform one-hot encoding for elements
    """
    def __init__(self, atom_types=[1, 6, 7, 8, 9]):
        super(AtomicOnehot, self).__init__()
        self.atom_types = atom_types

    def call(self, elems):
        output = tf.equal(tf.expand_dims(elems, 1),
                          tf.expand_dims(self.atom_types, 0))
        return output

class ANNOutput(tf.keras.layers.Layer):
    """ ANN Ouput layer

    Output atomic or molecular (system) properties
    """
    def __init__(self, out_pool):
        super(ANNOutput, self).__init__()
        self.out_pool = out_pool

    def call(self, tensors):
        ind_1, output = tensors

        if self.out_pool:
            out_pool = {'sum': tf.math.unsorted_segment_sum,
                        'max': tf.math.unsorted_segment_max,
                        'min': tf.math.unsorted_segment_min,
                        'avg': tf.math.unsorted_segment_mean,
            }[self.out_pool]
            output =  out_pool(output, ind_1[:,0],
                               tf.reduce_max(ind_1)+1)
        output = tf.squeeze(output, axis=1)

        return output

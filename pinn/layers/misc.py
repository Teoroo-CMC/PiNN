# -*- coding: utf-8 -*-

"""Misc. (Keras) Layers for Atomistic Neural Networks"""

import tensorflow as tf

class AtomicOnehot(tf.keras.layers.Layer):
    """One-hot embedding layer

    Given the atomic number of each atom ($Z_{i}$) and a list of specified
    element types ($Z_{c}$), returns:

    $$\mathbb{P}_{ic} = \delta_{Z_{i},Z_{c}}$$

    """
    def __init__(self, atom_types=[1, 6, 7, 8, 9]):
        """
        Args:
            atom_types (list of int): list of elements
        """
        super(AtomicOnehot, self).__init__()
        self.atom_types = atom_types

    def call(self, elems):
        """
        Args:
           elems (tensor): atomic indices of atoms, with shape `(n_atoms)`

        Returns:
           prop (tensor): atomic property tensor, with shape `(n_atoms, n_elems)`
        """
        prop = tf.equal(tf.expand_dims(elems, 1),
                          tf.expand_dims(self.atom_types, 0))
        return prop

class ANNOutput(tf.keras.layers.Layer):
    """ANN Ouput layer

    Output atomic or molecular (system) properties depending on `out_pool`

    $$
    \\begin{cases}
     \mathbb{P}^{\mathrm{out}}_i  &, \\textrm{if out_pool is False}\\\\
     \mathrm{pool}_i(\mathbb{P}^{\mathrm{out}}_i)  &, \\textrm{if out_pool}
    \end{cases}
    $$

    , where $\mathrm{pool}$ is a reducing operation specified with `out_pool`,
    it can be one of 'sum', 'max', 'min', 'avg'.

    """
    def __init__(self, out_pool):
        super(ANNOutput, self).__init__()
        self.out_pool = out_pool

    def call(self, tensors):
        """
        Args:
            tensors (list of tensor): ind_1 and output tensors

        Returns:
            output (tensor): atomic or per-structure predictions
        """
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

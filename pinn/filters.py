import tensorflow as tf
import numpy as np


class element_filter():
    def __init__(self, element_list=[1, 6, 7, 8, 9]):
        self.element_list = element_list

    def parse(self, atoms):
        p_mat = np.array([np.where(atoms.get_atomic_numbers() == e, 1, 0)
                          for e in self.element_list]).transpose()
        return p_mat

    def get_tensors(self, dtype, batch_size, n_max):
        p_in = tf.placeholder(
            dtype, shape=(batch_size, n_max, len(self.element_list)))
        p_mask = tf.reduce_sum(p_in, axis=-1, keepdims=True) > 0
        return p_in, p_mask


class f1_symm_func_filter():
    def __init__(self, rc=6.0, order=5):
        self.rc = rc
        self.order = order

    def get_tensors(self, d_mat):
        i_mask = (d_mat > 0) & (d_mat < self.rc)
        i_mat = tf.where(i_mask,
                         0.5*(tf.cos(np.pi*d_mat/self.rc)+1),
                         tf.zeros_like(d_mat))
        i_kernel = [i_mat ** (i+1) for i in range(self.order)]
        i_kernel = tf.concat(i_kernel, axis=-1)
        return i_kernel, i_mask


class f2_symm_func_filter():
    def __init__(self, rc=6.0, order=5):
        self.rc = rc
        self.order = order

    def get_tensors(self, d_mat):
        i_mask = (d_mat > 0) & (d_mat < self.rc)
        i_mat = tf.where(i_mask,
                         tf.tanh(1-d_mat/self.rc) ** 3,
                         tf.zeros_like(d_mat))

        i_kernel = [i_mat ** (i+1) for i in range(self.order)]
        i_kernel = tf.concat(i_kernel, axis=-1)
        return i_kernel, i_mask


default_p_filter = element_filter()
default_i_filter = f2_symm_func_filter()

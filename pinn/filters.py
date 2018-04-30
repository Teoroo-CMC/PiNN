import tensorflow as tf
import numpy as np


class element_filter():
    def __init__(self, element_list=[1, 6, 7, 8, 9]):
        self.element_list = element_list

    def parse(self, dataset, dtype):
        element_list = tf.expand_dims(tf.expand_dims(tf.constant(self.element_list), 0),0)

        elements = tf.expand_dims(dataset['p_in'], 2)
        p_mat = tf.cast(tf.equal(elements, element_list), dtype)
        dataset['p_in'] = p_mat
        return dataset

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

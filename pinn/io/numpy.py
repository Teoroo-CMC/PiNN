# -*- coding: utf-8 -*-
"""Numpy dataset"""

import numpy as np
import tensorflow as tf
from pinn.io.base import map_nested, split_list


def _numpy_generator(data_dict, subset):
    for i in subset:
        datum = {k: v[i] for k, v in data_dict.items()}
        yield datum


def load_numpy(data_dict, **kwargs):
    """Loads dataset from numpy array

    Args:
        data_dict: a dictionary of numpy arrays
        **kwargs: split options, see ``pinn.io.base.split_list``
    """
    dtypes = {
        k: (tf.int32 if v.dtype.kind in ('u', 'i') else tf.float32)
        for k, v in data_dict.items()}
    shapes = {k: v.shape[1:] for k, v in data_dict.items()}

    def generator_fn(subset): return tf.data.Dataset.from_generator(
        lambda: _numpy_generator(data_dict, subset), dtypes, shapes)
    subsets = split_list(list(range(data_dict['elems'].shape[0])), **kwargs)
    splitted = map_nested(generator_fn, subsets)
    return splitted

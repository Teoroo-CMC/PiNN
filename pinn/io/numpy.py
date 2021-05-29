# -*- coding: utf-8 -*-
"""Numpy dataset"""


def _numpy_generator(data_dict, subset):
    for i in subset:
        datum = {k: v[i] for k, v in data_dict.items()}
        yield datum


def load_numpy(dataset, splits=None, shuffle=True, seed=0):
    """Loads dataset from numpy array

    It is assumed that the numpy arrays contain structures with the
    same number of atoms, and stacked (with the same axis being the
    index of the structure).

    Args:
        dataset: a dictionary of numpy arrays
        splits (dict): key-val pairs specifying the ratio of subsets
        shuffle (bool): shuffle the dataset (only used when splitting)
        seed (int): random seed for shuffling
    """
    import numpy as np
    import tensorflow as tf
    from pinn.io.base import split_list

    dtypes = {
        k: ('int32' if v.dtype.kind in ('u', 'i') else tf.keras.backend.floatx())
        for k, v in dataset.items()}
    shapes = {k: v.shape[1:] for k, v in dataset.items()}

    def generator_fn(subset): return tf.data.Dataset.from_generator(
        lambda: _numpy_generator(dataset, subset), dtypes, shapes)

    indices = list(range(dataset['elems'].shape[0]))
    if splits is None:
        return generator_fn(indices)
    else:
        subsets = split_list(indices, splits=splits, shuffle=shuffle, seed=seed)
        splitted = {k: generator_fn(v) for k,v in subsets.items()}
        return splitted

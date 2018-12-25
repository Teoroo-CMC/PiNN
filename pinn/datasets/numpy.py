# -*- coding: utf-8 -*-
"""Numpy dataset
"""
from pinn.datasets.base import map_nested, split_list
import numpy as np
import tensorflow as tf
def _numpy_generator(data_dict, subset):
    for i in subset:
        datum = {k: v[i] for k,v in data_dict.items()}
        yield datum
    
def load_numpy_dataset(data_dict,
                        float_dtype=tf.float32, int_dtype=tf.int32,
                        split_ratio={'train': 8, 'test':1, 'vali':1},
                        shuffle=True, seed=0):
    """Loads dataset from numpy array
    """
    dtypes = {
        k:(int_dtype if v.dtype==np.integer else float_dtype)
        for k,v in data_dict.items()}
    shapes = {k:v.shape[1:] for k,v in data_dict.items()}
    generator_fn = lambda subset: tf.data.Dataset.from_generator(
        lambda: _numpy_generator(data_dict, subset), dtypes,shapes)
    subsets = split_list(list(range(data_dict['atoms'].shape[0])),
                         split_ratio, shuffle, seed)
    splitted = map_nested(generator_fn, subsets)
    return splitted
    
 
    
    

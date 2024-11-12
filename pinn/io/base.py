# -*- coding: utf-8 -*-
"""Basic functions for dataset loaders"""


def sparse_batch(batch_size, drop_remainder=False, num_parallel_calls=8,
                 atomic_props=['f_data', 'q_data', 'f_weights', 'oxidation']):
    """This returns a dataset operation that transforms single samples
    into sparse batched samples. The atomic_props must include all
    properties that are defined on an atomic basis besides 'coord' and
    'elems'.

    Args:
        drop_remainder (bool): option for padded_batch
        num_parallel_calls (int): option for map
        atomic_props (list): list of atomic properties
    """
    def sparsify(tensors):
        import tensorflow as tf
        atom_ind = tf.cast(tf.where(tensors['elems']), tf.int32)
        ind_1 = atom_ind[:, :1]
        ind_sp = tf.cumsum(tf.ones(tf.shape(ind_1), tf.int32))-1
        tensors['ind_1'] = ind_1
        elems = tf.gather_nd(tensors['elems'], atom_ind)
        coord = tf.gather_nd(tensors['coord'], atom_ind)
        tensors['elems'] = elems
        tensors['coord'] = coord
        # Optional
        for name in atomic_props:
            if name in tensors:
                tensors[name] = tf.gather_nd(tensors[name], atom_ind)
        return tensors
    def sparse_batch_op(dataset):
        shapes = {k:v.shape for k,v in dataset.element_spec.items()}
        return dataset.padded_batch(batch_size, shapes,
                             drop_remainder=drop_remainder).map(
                                 sparsify, num_parallel_calls)
    return sparse_batch_op


def split_list(data, splits={'train': 8, 'test': 2}, shuffle=True, seed=0):
    """
    Split the list according to a given ratio

    Args:
        data (list): a list of data to split
        splits (dict): a dictionary specifying the ratio of splits
        shuffle (bool): shuffle the list before
        seed (int): random seed used for shuffling

    Returns:
        a dictionary of the splitted list
    """
    import math, random
    data = data.copy() # work on a copy of the oridinal list
    n_tot = len(data)
    split_tot = float(sum([v for v in splits.values()]))
    n_split = {k:math.ceil(n_tot*v/split_tot) for k,v in splits.items()}
    if shuffle:
        random.seed(seed)
        random.shuffle(data)
    splitted = {}
    cnt = 0
    for k, v in n_split.items():
        splitted[k] = data[cnt:cnt+v]
        cnt += v
    return splitted


def list_loader(pbc=False, force=False, stress=False, ds_spec=None):
    """Decorator for building dataset loaders"""
    from functools import wraps
    import tensorflow as tf
    if ds_spec is None:
        ds_spec = {
            'elems': {'dtype':  'int32',   'shape': [None]},
            'coord': {'dtype':  'float', 'shape': [None, 3]},
            'e_data': {'dtype': 'float', 'shape': []},
        }
        if pbc:
            ds_spec['cell'] = {'dtype':  'float', 'shape': [3, 3]}
        if force:
            ds_spec['f_data'] = {'dtype':  'float', 'shape': [None, 3]}
        if stress:
            ds_spec['s_data'] = {'dtype':  'float', 'shape': [3, 3]}
    ds_spec = {k: tf.TensorSpec(**v) for k,v in ds_spec.items()}
    def decorator(func):
        @wraps(func)
        def data_loader(dataset, splits=None, shuffle=True, seed=0):
            """ Real data loader to use, with """
            dtype = tf.keras.backend.floatx()
            def _data_generator(dataset):
                for data in dataset:
                    yield func(data)
            def generator_fn(dataset): return tf.data.Dataset.from_generator(
                    lambda: _data_generator(dataset), output_signature=ds_spec)
            if splits is None:
                return generator_fn(dataset)
            else:
                subsets = split_list(dataset, splits=splits, shuffle=shuffle, seed=seed)
                splitted = {k: generator_fn(v) for k,v in subsets.items()}
                return splitted
        return data_loader
    return decorator


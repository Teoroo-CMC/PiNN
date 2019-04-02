"""Basic functions for datasets
"""

import random
import tensorflow as tf

class _datalist(list):
    """The same thing as list, but don't count in nested structure
    """
    pass

def map_nested(fn, nested):
    """Map fn to the nested structure
    """
    if isinstance(nested, dict):
        return {k: map_nested(fn, v) for k,v in nested.items()}
    if isinstance(nested, list) and type(nested) != _datalist:
        return [map_nested(fn, v) for v in nested]
    else:
        return fn(nested)

def flatten_nested(nested):
    """Retun a list of the nested elements
    """
    if isinstance(nested, dict):
        return sum([flatten_nested(v) for v in nested.values()],[])
    if isinstance(nested, list) and type(nested) != _datalist:
        return sum([flatten_nested(v) for v in nested], [])
    else:
        return [nested]

def split_list(data_list, split_ratio, shuffle=True, seed=None):
    """
    Split the list according to a given ratio

    Args:
        to_split (list): a list to split
        split_ratio: a nested (list and dict) of split ratio

    Returns:
        A nest structure of splitted data list
    """
    dummy = _datalist(data_list)
    if shuffle:
        random.seed(seed)
        random.shuffle(dummy)
    data_tot = len(dummy)
    split_tot = float(sum(flatten_nested(split_ratio)))
    get_split_num = lambda x:int(data_tot*x/split_tot)
    split_num = map_nested(get_split_num, split_ratio)
    def _pop_data(n):
        to_pop = dummy[:n]
        del dummy[:n]
        return _datalist(to_pop)
    splitted = map_nested(_pop_data, split_num)
    return splitted


def default_formater(n_atoms):
    """Default formater for datasets"""
    format_dict = {
    'atoms': {'dtype':  tf.int32,   'shape': [n_atoms]},
    'coord': {'dtype':  tf.float32, 'shape': [n_atoms, 3]},
    'e_data': {'dtype': tf.float32, 'shape': []}}
    return format_dict


def list_loader(formater=default_formater):
    from functools import wraps
    """Decorator for building dataset loaders"""
    def decorator(func):
        @wraps(func)
        def data_loader(data_list, n_atoms,
                        split_ratio={'train': 8, 'test':1, 'vali':1},
                        shuffle=True, seed=0, **kwargs):
            def _data_generator(data_list, n_atoms):
                for data in data_list:
                    yield func(data, n_atoms, **kwargs)
            format_dict = formater(n_atoms)
            dtypes = {k: v['dtype'] for k,v in format_dict.items()}
            shapes = {k: v['shape'] for k,v in format_dict.items()}
            generator_fn = lambda data_list: tf.data.Dataset.from_generator(
                lambda: _data_generator(data_list, n_atoms), dtypes, shapes)
            subsets = split_list(data_list, split_ratio, shuffle, seed)
            splitted = map_nested(generator_fn, subsets)
            return splitted
        return data_loader
    return decorator

# -*- coding: utf-8 -*-
"""The QM9 dataset

Url: http://quantum-machine.org/datasets/

@article{ramakrishnan2014quantum,
title={Quantum chemistry structures and properties of 134 kilo molecules},
author={Ramakrishnan, Raghunathan and Dral, Pavlo O and Rupp, Matthias and von Lilienfeld, O Anatole},
journal={Scientific Data},
volume={1},
year={2014},
publisher={Nature Publishing Group}
}
"""
import numpy as np
import tensorflow as tf
from ase.data import atomic_numbers
from pinn.datasets.base import map_nested, split_list

def _qm9_generator(filenames, n_atoms):
    for filename in filenames:
        f = open(filename)
        lines = f.readlines()
        atoms = [atomic_numbers[l.split()[0]] for l in lines[2:-3]]
        coord = [[i.replace('*^', 'E') for i in l.split()[1:4]]
                 for l in lines[2:-3]]
        e_data = float(lines[1].split()[12])
        f.close()

        to_pad = n_atoms-len(atoms)
        atoms = np.pad(atoms, [0,to_pad], 'constant')
        coord = np.pad(coord, [[0,to_pad], [0,0]], 'constant')

        data = {
            'atoms': atoms,
            'coord': coord,
            'e_data': e_data
        }
        yield data

def qm9_format(n_atoms=29,
               float_dtype=tf.float32,
               int_dtype=tf.int32):
    """Returns format dict for the QM9 dataset"""
    format_dict = {
        'atoms': {'dtype':  int_dtype,   'shape': [n_atoms],    'encode':False},
        'coord': {'dtype':  float_dtype, 'shape': [n_atoms, 3], 'encode':False},
        'e_data': {'dtype': float_dtype, 'shape': [],           'encode':False}}
    return format_dict

def load_QM9_dataset(filelist, n_atoms=29,
                     float_dtype=tf.float32, int_dtype=tf.int32,
                     split_ratio={'train': 8, 'test':1, 'vali':1},
                     shuffle=True, seed=0):
    """Loads the QM9 dataset

    Args:
        filelist (list):
        float_dtype:
        int_dtype:
        split_ratio, shuffle, seed:
            see ``pinn.datasets.base.split_list``
    """
    format_dict = qm9_format(n_atoms, float_dtype, int_dtype)
    dtypes = {k: v['dtype'] for k,v in format_dict.items()}
    shapes = {k: v['shape'] for k,v in format_dict.items()}
    generator_fn = lambda filelist: tf.data.Dataset.from_generator(
        lambda: _qm9_generator(filelist, n_atoms), dtypes, shapes)
    subsets = split_list(filelist, split_ratio, shuffle, seed)
    splitted = map_nested(generator_fn, subsets)
    return splitted

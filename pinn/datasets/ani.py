# -*- coding: utf-8 -*-
"""ANI-1, A data set of 20 million calculated off-equilibrium conformations
for organic molecules. (https://doi.org/10.6084/m9.figshare.c.3846712.v1)

Please cite the original paper when using this dataset.

The dataset is splitted by molecules while loading, 
meaninig the same molecule shows up in only one of the splitted datasets.
"""


import h5py
import numpy as np
import tensorflow as tf
from pinn.datasets.base import map_nested, split_list


def _ani_generator(sample_list, n_atoms):
    from ase.data import atomic_numbers as atomic_num
    for sample in sample_list:
        data = h5py.File(sample[0])[sample[1]]
        coord = data['coordinates'].value
        atoms = data['species'].value
        atoms = np.array([atomic_num[a.decode()] for a in atoms])
        atoms = np.tile(atoms[np.newaxis,:], [coord.shape[0],1])
        to_pad = n_atoms-atoms.shape[1]
        atoms = np.pad(atoms, [[0,0],[0,to_pad]], 'constant')
        coord = np.pad(coord, [[0,0],[0,to_pad], [0,0]], 'constant')
        e_data = data['energies'].value
        yield {'coord': coord, 'atoms': atoms, 'e_data': e_data}

        
def ani_format(n_atoms=26, float_dtype=tf.float32, int_dtype=tf.int32):
    """Returns format dict for the ANI-1 dataset"""
    format_dict = {
        'atoms': {'dtype':  int_dtype,   'shape': [n_atoms]},
        'coord': {'dtype':  float_dtype, 'shape': [n_atoms, 3]},
        'e_data': {'dtype': float_dtype, 'shape': []}}
    return format_dict


def load_ANI_dataset(filelist, n_atoms=None,
                     float_dtype=tf.float32, int_dtype=tf.int32,
                     split_ratio={'train': 8, 'test':1, 'vali':1},
                     shuffle=True, seed=0, cycle_length=4):
    """Loads the ANI-1 dataset

    Args:
        filelist (list): filenames of ANI-1 h5 files.
        n_atoms (int): max number of atoms, if this is not specified, .
            it will be inferred from the dataset, which is a bit slower.
        float_dtype: tensorflow datatype for float values.
        int_dtype: tensorflow datatype for integer values.
        split_ratio, shuffle, seed:
            see ``pinn.datasets.base.split_list``
    """
    format_dict = ani_format(n_atoms, float_dtype, int_dtype)
    dtypes = {k: v['dtype'] for k,v in format_dict.items()}
    shapes = {k: [None] + v['shape'] for k,v in format_dict.items()}
    # Load the list of samples
    max_n_atoms = 0
    sample_list = []
    for fname in filelist:
        store = h5py.File(fname)
        k1 = list(store.keys())[0]
        samples = store[k1]
        for k2 in samples.keys():
            sample_list.append((fname, '{}/{}'.format(k1, k2)))
            if n_atoms is None:
                max_n_atoms = max(max_n_atoms, samples[k2]['species'].shape[0])
    n_atoms = max_n_atoms if n_atoms is None else n_atoms
    # Generate dataset from sample list
    generator_fn = lambda samplelist: tf.data.Dataset.from_generator(
        lambda: _ani_generator(samplelist, n_atoms),
        dtypes, shapes).interleave(
            lambda x: tf.data.Dataset.from_tensor_slices(x),
            cycle_length=cycle_length)
    # Generate nested dataset
    subsets = split_list(sample_list, split_ratio, shuffle, seed)
    splitted = map_nested(generator_fn, subsets)
    return splitted    


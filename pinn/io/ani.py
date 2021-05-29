# -*- coding: utf-8 -*-
"""ANI-1, A data set of 20 million calculated off-equilibrium conformations
for organic molecules. (https://doi.org/10.6084/m9.figshare.c.3846712.v1)

Please cite the original paper when using this dataset.

The dataset is splitted by molecules while loading, 
meaninig the same molecule shows up in only one of the splitted datasets.
note that this scheme is not identical to the original ANI-1 paper.
"""




def _ani_generator(sample_list):
    from ase.data import atomic_numbers as atomic_num
    for sample in sample_list:
        data = h5py.File(sample[0])[sample[1]]
        coord = data['coordinates'].value
        elems = data['species'].value
        elems = np.array([atomic_num[e.decode()] for e in elems])
        elems = np.tile(elems[np.newaxis, :], [coord.shape[0], 1])
        e_data = data['energies'].value
        yield {'coord': coord, 'elems': elems, 'e_data': e_data}


def load_ani(filelist, split=False, shuffle=True, seed=0, cycle_length=4):
    """Loads the ANI-1 dataset

    Args:
        filelist (list): filenames of ANI-1 h5 files.
        split (dict): key-val pairs specifying the ratio of subsets
        shuffle (bool): shuffle the dataset (only used when splitting)
        seed (int): random seed for shuffling
        cycle_length (int): number of parallel threads to read h5 file
    """
    import h5py
    import numpy as np
    import tensorflow as tf
    from pinn.io.base import split_list
    ds_spec = {
        'elems': {'dtype':  tf.int32,   'shape': [None]},
        'coord': {'dtype':  tf.keras.backend.floatx(), 'shape': [None, 3]},
        'e_data': {'dtype': tf.keras.backend.floatx(), 'shape': []}}
    # Load the list of samples
    sample_list = []
    for fname in filelist:
        store = h5py.File(fname)
        k1 = list(store.keys())[0]
        samples = store[k1]
        for k2 in samples.keys():
            sample_list.append((fname, '{}/{}'.format(k1, k2)))
    # Generate dataset from sample list

    def generator_fn(samplelist): return tf.data.Dataset.from_generator(
            lambda: _ani_generator(samplelist), output_signature=ds_spec).interleave(
            lambda x: tf.data.Dataset.from_tensor_slices(x),
            cycle_length=cycle_length)
    # Generate nested dataset
    subsets = split_list(sample_list, split=split, shuffle=shuffle, seed=0)
    splitted = map_nested(generator_fn, subsets)
    return splitted

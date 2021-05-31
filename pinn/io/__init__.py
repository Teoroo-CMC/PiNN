# -*- coding: utf-8 -*-
from pinn.io.base import list_loader, sparse_batch
from pinn.io.tfr import load_tfrecord, write_tfrecord
from pinn.io.ase import load_ase
from pinn.io.runner import load_runner
from pinn.io.qm9 import load_qm9
from pinn.io.ani import load_ani
from pinn.io.cp2k import load_cp2k
from pinn.io.numpy import load_numpy

def load_ds(dataset, fmt='auto', splits=False, shuffle=True, seed=0, **kwargs):
    """This loader tries to guess the format when dataset is a string:

    - `load_tfrecoard` if it ends with '.yml'
    - `load_runner` if it ends with '.data'
    - try to load it with `load_ase`

    If the `fmt` is specified, the loader will use a corresponsing dataset loader.

    Args:
        dataset: dataset a file or input for a loader according to `fmt`
        fmt (str): dataset format, see avialable formats.
        splits (dict): key-val pairs specifying the ratio of subsets
        shuffle (bool): shuffle the dataset (only used when splitting)
        seed (int): random seed for shuffling
        **kwargs: extra arguments to loaders
    """
    loaders = {'tfr':    load_tfrecord,
               'runner': load_runner,
               'ase':    load_ase,
               'qm9':    load_qm9,
               'ani':    load_ani,
               'cp2k':   load_cp2k}
    if fmt=='auto':
        if dataset.endswith('.yml'):
            return load_tfrecord(dataset, splits=splits, shuffle=shuffle, seed=seed)
        if dataset.endswith('.data'):
            return load_runner(dataset, splits=splits, shuffle=shuffle, seed=seed)
        else:
            return load_ase(dataset, splits=splits, shuffle=shuffle, seed=seed)
    else:
        return loaders['fmt'](dataset, splits=splits, shuffle=shuffle, seed=seed, **kwargs)

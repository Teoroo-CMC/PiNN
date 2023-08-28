# -*- coding: utf-8 -*-


def _qm9_spec(label_map):
    """Returns format dict for the QM9 dataset"""
    ds_spec = {
        'elems': {'dtype':  'int32',   'shape': [None]},
        'coord': {'dtype':  'float', 'shape': [None, 3]}}
    for k, v in label_map.items():
        ds_spec[k] = {'dtype':  'float', 'shape': []}
    return ds_spec


def load_qm9(flist, label_map={'e_data': 'U0'}, splits=None, shuffle=True, seed=0):
    """Loads the QM9 dataset

    QM9 provides a variety of labels, but typically we are only
    training on one target, e.g. U0. A ``label_map`` option is
    offered to choose the output dataset structure, by default, it
    only takes "U0" and maps that to "e_data",
    i.e. `label_map={'e_data': 'U0'}`.

    Other available labels are

    ```python
    ['tag', 'index', 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo',
     'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
    ```

    Desciptions of those tags can be found in QM9's description file.

    Args:
        flist (list): list of QM9-formatted data files.
        splits (dict): key-val pairs specifying the ratio of subsets
        shuffle (bool): shuffle the dataset (only used when splitting)
        seed (int): random seed for shuffling
        label_map (dict): dictionary mapping labels to output datasets
    """
    import numpy as np
    import tensorflow as tf
    from pinn.io.base import list_loader
    from ase.data import atomic_numbers

    _labels = ['tag', 'index', 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap',
               'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
    _label_ind = {k: i for i, k in enumerate(_labels)}

    @list_loader(ds_spec=_qm9_spec(label_map))
    def _qm9_loader(fname):
        with open(fname) as f:
            lines = f.readlines()
        elems = [atomic_numbers[l.split()[0]] for l in lines[2:-3]]
        coord = [[i.replace('*^', 'E') for i in l.split()[1:4]]
                 for l in lines[2:-3]]
        elems = np.array(elems, np.int32)
        coord = np.array(coord, float)
        data = {'elems': elems, 'coord': coord}
        for k, v in label_map.items():
            data[k] = float(lines[1].split()[_label_ind[v]])
        return data
    return _qm9_loader(flist, splits=splits, shuffle=shuffle, seed=seed)

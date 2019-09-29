# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from ase.data import atomic_numbers
from pinn.io.base import list_loader


def _qm9_format(label_map):
    """Returns format dict for the QM9 dataset"""
    format_dict = {
        'elems': {'dtype':  tf.int32,   'shape': [None]},
        'coord': {'dtype':  tf.float32, 'shape': [None, 3]}}
    for k, v in label_map.items():
        format_dict[k] = {'dtype':  tf.float32, 'shape': []}
    return format_dict


def load_qm9(flist, label_map={'e_data': 'U0'}, **kwargs):
    """Loads the QM9 dataset

    QM9 provides a variety of labels, but in a typical usage we 
    are only training on one target, e.g. U0.
    Therefore, a label_map option is offered to choose the output
    dataset structure, by default, it only takes "U0" and maps that
    to "e_data", e.g. label_map={'e_data': 'U0'}.

    Other avaiable labels are::

        ['tag', 'index', 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo',
         'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']

    Desciptions about those tags can be found in QM9's description file.

    Args:
        flist (list): list of QM9-formatted data files.
        label_map (dict): dictiionary
        **kwargs: split options, see ``pinn.io.base.split_list``
    """
    _labels = ['tag', 'index', 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap',
               'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
    _label_ind = {k: i for i, k in enumerate(_labels)}

    @list_loader(format_dict=_qm9_format(label_map))
    def _qm9_loader(fname):
        with open(fname) as f:
            lines = f.readlines()
        elems = [atomic_numbers[l.split()[0]] for l in lines[2:-3]]
        coord = [[i.replace('*^', 'E') for i in l.split()[1:4]]
                 for l in lines[2:-3]]
        elems = np.array(elems, np.int32)
        coord = np.array(coord, np.float32)
        data = {'elems': elems, 'coord': coord}
        for k, v in label_map.items():
            data[k] = float(lines[1].split()[_label_ind[v]])
        return data
    return _qm9_loader(flist, **kwargs)

# -*- coding: utf-8 -*-
"""Helper functions for tests"""

import os
import numpy as np
from helpers import *

this_dir = os.path.dirname(os.path.abspath(__file__))

# Example traivial dataset for testing purpose,
# In all trivial datasets, there are three atoms,
# sitting at (0, 0, 0), (1, 0, 0) and (1, 1, 0)


def get_trivial_qm9_ds():
    from glob import glob
    from pinn.io import load_qm9
    flist = glob('{}/examples/*.xyz'.format(this_dir))
    dataset = load_qm9(flist, split=1)
    return dataset


def get_trivial_runner_ds():
    from pinn.io import load_runner
    fname = '{}/examples/trivial.data'.format(this_dir)
    dataset = load_runner(fname, split=1)
    return dataset


def get_trivial_numpy():
    trivial_data = {
        'coord': np.array(
            [[0.0, 0.0, 0.0],
             [1.0, 0.0, 0.0],
             [1.0, 1.0, 0.0]], np.float32),
        'elems': np.array([1, 8, 1], np.int32),
        'cell': np.array(
            [[100.0, 0.0, 0.0],
             [0.0, 100.0, 0.0],
             [0.0, 0.0, 100.0]], np.float32),
        'e_data': np.array(11.0)
    }
    return trivial_data


def get_trivial_numpy_ds():
    import numpy as np
    from pinn.io import load_numpy
    data = get_trivial_numpy()
    data = {k: np.expand_dims(v, axis=0) for k, v in data.items()}
    dataset = load_numpy(data, split=1)
    return dataset


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from pinn.io import load_numpy, sparse_batch
from ase import Atoms
import pytest

def mock_data(n_data, n_batch):
    from ase.calculators.lj import LennardJones
    atoms = Atoms('H3', positions=[[0, 0, 0], [0, 1, 0], [1, 1, 0]])
    atoms.calc = LennardJones(rc=5.0)
    coord, elems, e_data, f_data = [], [], [], []
    for x_a in np.linspace(-5, 0, n_data):
        atoms.positions[0, 0] = x_a
        coord.append(atoms.positions.copy())
        elems.append(atoms.numbers)
        e_data.append(atoms.get_potential_energy())
        f_data.append(atoms.get_forces())

    data = {
        'coord': np.array(coord),
        'elems': np.array(elems),
        'e_data': np.array(e_data),
        'f_data': np.array(f_data)
    }
    dataset = load_numpy(data).apply(sparse_batch(n_batch))
    return dataset

@pytest.fixture
def mocked_data():
    return mock_data(100, 10)
#!/usr/bin/env python

from pinn.io.base import list_loader

def _ase_spec(atoms):
    """Guess dataset format from an ASE atoms"""
    ds_spec = {
        'elems': {'dtype':  'int32', 'shape': [None]},
        'coord': {'dtype':  'float', 'shape': [None, 3]}}

    if atoms.pbc.any():
        ds_spec['cell'] = {'dtype': 'float', 'shape': [3, 3]}

    try:
        atoms.get_potential_energy()
        ds_spec['e_data'] = {'dtype': 'float', 'shape': []}
    except:
        pass

    try:
        atoms.get_forces()
        ds_spec['f_data'] = {'dtype': 'float', 'shape': [None, 3]}
    except:
        pass

    try:
        atoms.get_charges()
        ds_spec['q_data'] = {'dtype': 'float', 'shape': [None]}
    except:
        pass

    try:
        atoms.get_dipole_moment()
        ds_spec['d_data'] = {'dtype': 'float', 'shape': [3]}
    except:
        pass

    return ds_spec

def load_ase(dataset, splits=None, shuffle=True, seed=0):
    """
    Loads a ASE trajectory

    Args:
        dataset (str or ase.io.trajectory): a filename or trajectory
        splits (dict): key-val pairs specifying the ratio of subsets
        shuffle (bool): shuffle the dataset (only used when splitting)
        seed (int): random seed for shuffling
    """
    from ase.io import read

    if isinstance(dataset, str):
        dataset = read(dataset, index=':')

    ds_spec = _ase_spec(dataset[0])
    @list_loader(ds_spec=ds_spec)
    def _ase_loader(atoms):
        datum = {
            'elems': atoms.numbers,
            'coord': atoms.positions,
        }
        if 'cell' in ds_spec:
            datum['cell'] = atoms.cell[:]

        if 'e_data' in ds_spec:
            datum['e_data'] = atoms.get_potential_energy()

        if 'f_data' in ds_spec:
            datum['f_data'] = atoms.get_forces()

        if 'q_data' in ds_spec:
            datum['q_data'] = atoms.get_charges()

        if 'd_data' in ds_spec:
            datum['d_data'] = atoms.get_dipole_moment()
        return datum

    return _ase_loader(dataset, splits=splits, shuffle=shuffle, seed=seed)

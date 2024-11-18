from pathlib import Path


def load_deepmd(fdict_or_fpath, type_map=None, pbc=True, shuffle=True, seed=0):

    """This is loader for deepmd input data. It takes a dict of key and file path or a directory path which contains the data files. If type_map is provided, it will be used to convert the type id to atomic numbers.

    | key        | data source         | provides         |
    |------------|---------------------|------------------|
    | `coord`    | `path/coord.raw`      | `coord`  |
    | `force`    | `path/force.raw`      | `f_data`      |
    | `energy`   | `path/energy.raw`      | `e_data`      |
    | `virial`   | `path/virial.raw`      | `s_data`      |
    | `box`     | `path/box.raw` | `cell`           |
    | `type`    | `path/type.raw` | `elems`           |

    Args:
        files (dict | Path | str): input files
        shuffle (bool): shuffle the dataset (only used when splitting)
        seed (int): random seed for shuffling
    """
    if isinstance(fdict_or_fpath, (Path, str)):
        fdict = {}
        for key in ['coord', 'force', 'energy', 'virial', 'box', 'elems']:
            fdict[key] = Path(fdict_or_fpath) / f'{key}.raw'
    else:
        assert all([key in fdict_or_fpath for key in ['coord', 'force', 'energy', 'virial', 'box', 'elems']])
        fdict = fdict_or_fpath

    from ase.data import chemical_symbols
    import numpy as np
    from pinn.io import list_loader

    coord = np.loadtxt(fdict['coord'])
    force = np.loadtxt(fdict['force'])
    energy = np.loadtxt(fdict['energy'])
    # stress = np.loadtxt(fdict['virial'])
    cell = np.loadtxt(fdict['cell'])
    elem = np.loadtxt(fdict['elems'], dtype=int)

    if type_map is not None:
        if isinstance(type_map, (bool)):
            type_map_path = Path(fdict_or_fpath) / 'type_map.raw'

        elif isinstance(type_map, (Path, str)):
            type_map_path = Path(type_map)
    if type_map_path.exists():
        with open(type_map_path, 'r') as f:
            # assume type.raw is incremental integers starting from 0
            type_map = {chemical_symbols.index(line.strip()): i for i, line in enumerate(f)}

    for k, v in type_map.items():
        elem[elem == v] = k

    data = []
    # DeePMD .raw files use units of Å and eV. [https://docs.deepmodeling.com/projects/deepmd/en/latest/data/system.html]
    for i in range(len(coord)):
        data.append({
            'coord': coord[i],
            'f_data': force[i],
            'e_data': energy[i],
            # 's_data': stress[i],
            'cell': cell[i],
            'elems': elem
        })

    ds_spec = {
    'elems':  {'dtype':  'int32','shape': [None]},
    'cell':   {'dtype': 'float', 'shape': [3, 3]},
    'coord':  {'dtype':  'float','shape': [None, 3]},
    'e_data': {'dtype': 'float', 'shape': []},
    'f_data': {'dtype': 'float', 'shape': [None, 3]},
    # 's_data': {'dtype': 'float', 'shape': [3, 3]},
}

    @list_loader(ds_spec=ds_spec, pbc=pbc)
    def _frame_loader(datum):
        return datum
    
    return _frame_loader(data, shuffle=shuffle, seed=seed)

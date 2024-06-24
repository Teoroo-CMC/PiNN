# -*- coding: utf-8 -*-
from ase.units import create_units

# This is to follow the CP2K standard to use CODATA 2006, which differs from the
# the defaults of ASE (as of ASE ver 3.23 and CP2K v2022.1, Sep 2022)
units = create_units("2006")

def _cell_dat_indexer(files):
    import numpy as np
    if not isinstance(files['cell_dat'], str):
        return [files['cell_dat']]
    else:
        # the line start with # will be ignored by np zzy 20240324
        arrCells = np.loadtxt(files['cell_dat'], usecols=(2,3,4))
        arrLocs = np.arange(0,arrCells.shape[0],1)
        listLocs = arrLocs.tolist()
        indexes = list(zip([files['cell_dat']] * len(listLocs), listLocs))
        return indexes

def _cell_dat_loader(index):
    import numpy as np
    fname, loc = index
    data = []
    arrCellXs = np.loadtxt(fname, usecols=(2, 3, 4))
    data.append(arrCellXs[loc].tolist())
    arrCellYs = np.loadtxt(fname, usecols=(5, 6, 7))
    data.append(arrCellYs[loc].tolist())
    arrCellZs = np.loadtxt(fname, usecols=(8, 9, 10))
    data.append(arrCellZs[loc].tolist())
    return {'cell': np.array(data, float)}

def _stress_indexer(files):
    import mmap, re
    f = open(files['out'], 'r')
    m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    locs = [match.span()[0] for match in
            re.finditer(br' STRESS\| Analytical stress tensor \[GPa\]', m)]
    indexes = list(zip([files['out']]*len(locs), locs))
    f.close()
    return indexes

def _stress_loader(index):
    import numpy as np
    data = []
    fname, loc = index
    f = open(fname, 'r')
    f.seek(loc)
    assert f.readline().startswith(
        " STRESS| Analytical stress tensor [GPa]"
    ) & f.readline().startswith(
        " STRESS|                        x"
    ), "Unknown format of CP2K log, aborting"
    for i in range(3):
        l = f.readline().strip()
        data.append(l.split()[2:])
    f.close()
    return {"s_data": -np.array(data, float) * units["GPa"]}

def _energy_indexer(files):
    import mmap, re
    f = open(files['out'], 'r')
    regex = r'ENERGY\|\ Total FORCE_EVAL.*:\s*([-+]?\d*\.?\d*)'
    energies = [float(e) * units["Hartree"] for e in re.findall(regex, f.read())]
    f.close()
    nPrintStep = 1 # should change the value based on your own cp5k setting
    print("Note: the snapshotsteps in CP2K is set as %d"%(nPrintStep))
    return energies[::nPrintStep]

def _energy_loader(energy):
    return {'e_data': energy}

def _force_indexer(files):
    import mmap, re
    f = open(files['out'], 'r')
    m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    locs = [match.span()[0] for match in
            re.finditer(br' ATOMIC FORCES in \[a.u.\]', m)]
    indexes = list(zip([files['out']]*len(locs), locs))
    f.close()
    return indexes

def _force_loader(index):
    import numpy as np
    fname, loc = index

    data = []
    f = open(fname, "r")
    f.seek(loc)
    assert (
        f.readline().startswith(" ATOMIC FORCES in [a.u.]")
        & f.readline().startswith("\n")
        & f.readline().startswith(" # Atom   Kind   Element")
    ), "Unknown format of CP2K log, aborting"
    l = f.readline().strip()
    while not l.startswith("SUM OF"):
        data.append(l.split()[3:])
        l = f.readline().strip()
    f.close()

    return {"f_data": np.array(data, float) * units["Hartree"] / units["Bohr"]}

def _coord_indexer(files):
    import mmap,re
    f = open(files['coord'], 'r')
    first_line = f.readline(); f.seek(0);
    regex = str.encode('(^|\n)'+first_line[:-1]+'(\r\n|\n)')
    m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    locs = [match.span()[-1] for match in
            re.finditer(regex, m)]
    indexes = list(zip([files['coord']]*len(locs), locs))
    f.close()
    return indexes

def _coord_loader(index):
    import numpy as np
    from ase.data import atomic_numbers
    fname, loc = index
    elems = []
    coord = []
    f = open(fname, 'r')
    f.seek(loc)
    f.readline()
    while True:
        line = f.readline().split()
        if len(line) <= 1:
            break
        elems.append(atomic_numbers[line[0]])
        coord.append(line[1:4])
    f.close()
    return {'elems': np.array(elems, int),
            'coord': np.array(coord, float)}


indexers = {'force': _force_indexer,
            'energy': _energy_indexer,
            'stress': _stress_indexer,
            'coord': _coord_indexer,
            'cell_dat': _cell_dat_indexer}

loaders = {'force': _force_loader,
           'energy': _energy_loader,
           'stress': _stress_loader,
           'coord': _coord_loader,
           'cell_dat': _cell_dat_loader}

formats = {
    'elems':  {'dtype':  'int32','shape': [None]},
    'cell':   {'dtype': 'float', 'shape': [3, 3]},
    'coord':  {'dtype':  'float','shape': [None, 3]},
    'e_data': {'dtype': 'float', 'shape': []},
    'f_data': {'dtype': 'float', 'shape': [None, 3]},
    's_data': {'dtype': 'float', 'shape': [3, 3]},
}

provides = {
    'force': ['f_data'],
    'energy': ['e_data'],
    'stress': ['s_data'],
    'coord':  ['coord', 'elems'],
    'cell_dat': ['cell']
}

def _gen_list(files, keys):
    all_list = {k: [] for k in keys}
    for i, file in enumerate(files):
        new_list = {}
        for key in keys:
            new_list[key] = indexers[key](file)
        # Check each set of data have the same size
        assert len(set([len(v) for v in new_list.values()]))<=1
        print(f'\rIndexing: {i+1}/{len(files)}', end='')
        for k in keys:
            all_list[k] += new_list[k]
    print()
    return all_list

def load_cp2k(files, keys, splits=None, shuffle=True, seed=0):
    """This is a experimental loader for CP2K data

    It takes data from different sources, the CP2K output file and dat files,
    which will be specified in the files dictionary. A list of "keys" is used to
    specify the data to read and where it is read from.

    | key        | data source         | provides         |
    |------------|---------------------|------------------|
    | `force`    | `files['out']`      | `f_data`         |
    | `energy`   | `files['out']`      | `e_data`         |
    | `stress`   | `files['out']`      | `coord`, `elems` |
    | `cell_dat` | `files['cell_dat']` | `cell`           |

    Args:
        files (dict): input files
        keys (list): data to read
        splits (dict): key-val pairs specifying the ratio of subsets
        shuffle (bool): shuffle the dataset (only used when splitting)
        seed (int): random seed for shuffling
    """
    from pinn.io import list_loader
    ds_spec = {}
    for key in keys:
        for name in provides[key]:
            ds_spec.update({name:formats[name]})

    all_list = _gen_list(files, keys)

    @list_loader(ds_spec=ds_spec)
    def _frame_loader(i):
        results = {}
        for k,v in all_list.items():
            results.update(loaders[k](v[i]))
        return results

    return _frame_loader(list(range(len(all_list['coord']))),
                         splits=splits, shuffle=shuffle, seed=seed)

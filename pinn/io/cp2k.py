# -*- coding: utf-8 -*-

def _cell_dat_indexer(files):
    if isinstance(files['cell_dat'], str):
        return [files['cell_dat']]
    else:
        return files['cell_dat']

def _cell_dat_loader(index):
    import numpy as np
    return {'cell': np.loadtxt(index, usecols=(1,2,3))}

def _stress_indexer(files):
    import mmap, re
    f = open(files['out'], 'r')
    m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    locs = [match.span()[0] for match in
            re.finditer(b'STRESS TENSOR \[GPa\]', m)]
    indexes = list(zip([files['out']]*len(locs), locs))
    f.close()
    return indexes

def _stress_loader(index):
    import numpy as np
    fname, loc = index
    f = open(fname, 'r')
    data = []
    f.seek(loc)
    [f.readline() for i in range(3)]
    for i in range(3):
        l = f.readline().strip()
        data.append(l.split()[1:])
    unit = -1e9*2.2937e17*1e-30 # GPa -> Hartree/Ang^3
    f.close()
    return {'s_data': np.array(data, np.float)*unit}

def _energy_indexer(files):
    import mmap, re
    f = open(files['out'], 'r')
    regex = r'ENERGY\|\ Total FORCE_EVAL.*:\s*([-+]?\d*\.?\d*)'
    energies = [float(e) for e in re.findall(regex, f.read())]
    f.close()
    return energies

def _energy_loader(energy):
    return {'e_data': energy}

def _force_indexer(files):
    import mmap, re
    f = open(files['out'], 'r')
    m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    locs = [match.span()[0] for match in
            re.finditer(b'ATOMIC FORCES in', m)]
    indexes = list(zip([files['out']]*len(locs), locs))
    f.close()
    return indexes

def _force_loader(index):
    import numpy as np
    bohr2ang = 0.5291772109
    fname, loc = index
    f = open(fname, 'r')
    data = []
    f.seek(loc)
    [f.readline() for i in range(3)]
    l = f.readline().strip()
    while not l.startswith('SUM OF'):
        data.append(l.split()[3:])
        l = f.readline().strip()
    f.close()
    return {'f_data': np.array(data, np.float)/bohr2ang}

def _coord_indexer(files):
    import mmap
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
    return {'elems': np.array(elems, np.float),
            'coord': np.array(coord, np.float)}


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

def load_cp2k(files, keys, **kwargs):
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

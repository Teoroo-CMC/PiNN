# -*- coding: utf-8 -*-
import re
import numpy as np
import tensorflow as tf
from ase.data import atomic_numbers
from pinn.io import list_loader

def _cell_dat_indexer(files):
    if isinstance(files['cell_dat'], str):
        return [files['cell_dat']]
    else:
        return files['cell_dat']

def _cell_dat_loader(index):
    import numpy as np
    return {'cell': np.loadtxt(index, usecols=(1,2,3))}

def _stress_indexer(files):
    import mmap
    f = open(files['out'], 'r')
    m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    locs = [match.span()[0] for match in
            re.finditer(b'STRESS TENSOR \[GPa\]', m)]
    indexes = list(zip([files['out']]*len(locs), locs))
    f.close()
    return indexes

def _stress_loader(index):
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
    import mmap
    f = open(files['out'], 'r')
    regex = r'ENERGY\|\ Total FORCE_EVAL.*:\s*([-+]?\d*\.?\d*)'
    energies = [float(e) for e in re.findall(regex, f.read())]
    f.close()
    return energies

def _energy_loader(energy):
    return {'e_data': energy}

def _force_indexer(files):
    import mmap
    f = open(files['out'], 'r')
    m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    locs = [match.span()[0] for match in
            re.finditer(b'ATOMIC FORCES in', m)]
    indexes = list(zip([files['out']]*len(locs), locs))
    f.close()
    return indexes

def _force_loader(index):
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
    'elems': {'dtype':  tf.int32,   'shape': [None]},
    'coord': {'dtype':  tf.float32, 'shape': [None, 3]},
    'cell': {'dtype': tf.float32, 'shape': [3, 3]},
    'e_data': {'dtype': tf.float32, 'shape': []},
    'f_data': {'dtype': tf.float32, 'shape': [None, 3]},
    's_data': {'dtype': tf.float32, 'shape': [3, 3]},
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
    format_dict = {}
    for key in keys:
        for name in provides[key]:
            format_dict.update({name:formats[name]})

    all_list = _gen_list(files, keys)

    @list_loader(format_dict=format_dict)
    def _frame_loader(i):
        results = {}
        for k,v in all_list.items():
            results.update(loaders[k](v[i]))
        return results

    return _frame_loader(list(range(len(all_list['coord']))), **kwargs)

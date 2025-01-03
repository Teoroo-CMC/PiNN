# -*- coding: utf-8 -*-
"""A RuNNer data loader
RuNNer data has the format
    begin
    lattice float float float
    lattice float float float
    lattice float float float
    atom floatcoordx floatcoordy floatcoordz int_atom_symbol floatq 0  floatforcex floatforcey floatforcez
    atom 1           2           3           4               5      6  7           8           9
    energy float
    charge float
    comment arbitrary string
    end
The order of the lines within the begin/end block are arbitrary.
Coordinates, charges, energies and forces are all in atomic units.
Originally written by: Matti Hellström
Modified by: Yunqi Shao [yunqi.shao@kemi.uu.se]
[yqshao@2020-06-06]: Added stress (not a standard RuNNer keyword)
"""

from pinn.io import list_loader

runner_spec = {
    'elems':     {'dtype': 'int32',   'shape': [None]},
    'cell':      {'dtype': 'float', 'shape': [3, 3]},
    'coord':     {'dtype': 'float', 'shape': [None, 3]},
    'e_data':    {'dtype': 'float', 'shape': []},
    'f_data':    {'dtype': 'float', 'shape': [None, 3]},
    'q_data':    {'dtype': 'float', 'shape': [None]},
    's_data':    {'dtype': 'float', 'shape': [3, 3]},
    'e_weight':  {'dtype': 'float', 'shape': []},
    'f_weights': {'dtype': 'float', 'shape': [None, 3]},
}


@list_loader(ds_spec=runner_spec)
def _frame_loader(frame):
    import numpy as np
    from ase.data import atomic_numbers
    fname, pos = frame
    bohr2ang = 0.5291772109
    with open(fname) as f:
        f.seek(pos)
        elems = []
        coord = []
        q_data = []
        f_data = []
        s_data = []
        f_weights = []
        e_weight = 1.0
        cell = []
        line = f.readline().strip()
        while line:
            splitline = line.split()
            if splitline[0] == "atom":
                elems.append(atomic_numbers[splitline[4]])
                coord.append([float(splitline[1])*bohr2ang,
                              float(splitline[2])*bohr2ang,
                              float(splitline[3])*bohr2ang])
                q_data.append(float(splitline[5]))
                f_data.append([float(splitline[7])/bohr2ang,
                               float(splitline[8])/bohr2ang,
                               float(splitline[9])/bohr2ang])
                temp_f_weights = [1.0, 1.0, 1.0]
                if len(splitline) >= 13:
                    temp_f_weights = [float(splitline[10]),
                                      float(splitline[11]),
                                      float(splitline[12])]
                f_weights.append(temp_f_weights)
            elif splitline[0] == "energy":
                e_data = float(splitline[1])
            elif splitline[0] == "energy_weight":
                e_weight = float(splitline[1])
            elif splitline[0] == "lattice":
                cell.append([float(splitline[1])*bohr2ang,
                             float(splitline[2])*bohr2ang,
                             float(splitline[3])*bohr2ang])
            elif splitline[0] == "stress":
                s_data.append([float(splitline[1])*bohr2ang**3,
                               float(splitline[2])*bohr2ang**3,
                               float(splitline[3])*bohr2ang**3])
            elif line == "end":
                break
            line = f.readline().strip()

        if s_data == []:
            s_data = np.zeros([3,3], float)
        else:
            s_data = np.array(s_data, float)
        elems = np.array(elems, np.int32)
        coord = np.array(coord, float)
        f_data = np.array(f_data, float)
        f_weights = np.array(f_weights, float)
        cell = np.array(cell, float)
        q_data = np.array(q_data, float)
        data = {
            'elems': elems,
            'coord': coord,
            'e_data': e_data,
            'f_data': f_data,
            'q_data': q_data,
            's_data': s_data,
            'e_weight': e_weight,
            'f_weights': f_weights,
            'cell': cell
        }
        return data


def _gen_frame_list(fname):
    import re
    i = 0
    frame_list = []
    with open(fname) as f:
        for l in f:
            if re.match('.*begin', l):
                frame_list.append((fname, i))
            i += len(l)
    return frame_list


def load_runner(flist, splits=None, shuffle=True, seed=0):
    """
    Loads runner formatted trajectory. Bohr is converted to Angstrom automatically.

    Args:
        flist (str): one or a list of runner formatted trajectory(s)
        splits (dict): key-val pairs specifying the ratio of subsets
        shuffle (bool): shuffle the dataset (only used when splitting)
        seed (int): random seed for shuffling
    """
    if isinstance(flist, str):
        flist = [flist]
    frame_list = []
    for fname in flist:
        frame_list += _gen_frame_list(fname)
    return _frame_loader(frame_list, splits=splits, shuffle=shuffle, seed=seed)


def write_runner(fname, dataset):
    from ase.data import chemical_symbols
    bohr2ang = 0.5291772109
    lines = []
    for idx, data in enumerate(dataset):
        lines += ['begin\n', f'comment runner dataset generated by PiNN\n']
        c = data['cell']/bohr2ang
        for i in range(3):
            lines.append(f'lattice {c[i,0]:14.6e} {c[i,1]:14.6e} {c[i,2]:14.6e}\n')
        if 's_data' in data:
            s = data['s_data']/bohr2ang**3
            for i in range(3):
                lines.append(f'stress  {s[i,0]:14.6e} {s[i,1]:14.6e} {s[i,2]:14.6e}\n')
        for e, c, f in zip(data['elems'], data['coord']/bohr2ang, data['f_data']*bohr2ang):
            lines.append(f'atom     {c[0]:14.6e} {c[1]:14.6e} {c[2]:14.6e}  '
                         f'{chemical_symbols[e]}  '
                         f'0.0    0.0   {f[0]:14.6e} {f[1]:14.6e} {f[2]:14.6e} \n')
        lines.append(f'energy  {data["e_data"]:14.6e}\n')
        lines.append('charge 0.0\n')
        lines.append('end\n')
        print(f'\r Loading {idx} ...', end='')
    with open(fname, 'w') as file:
        file.writelines(lines)
    print('Done!')

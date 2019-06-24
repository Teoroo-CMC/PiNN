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

The order of the lines within the begin/end block are arbitrary
The coordinates are given in bohr, the charges in atomic units, the energy in Ha, and the force components in Ha/bohr

Originally written by: Matti Hellström
Modified by: Yunqi Shao [yunqi.shao@kemi.uu.se]    
"""
import re
import tensorflow as tf
import numpy as np
from pinn.io import list_loader
from ase.data import atomic_numbers

runner_format = {
    'cell': {'dtype':  tf.float32, 'shape': [3,3]},
    'elems': {'dtype':  tf.int32,   'shape': [None]},
    'coord': {'dtype':  tf.float32, 'shape': [None, 3]},
    'e_data': {'dtype': tf.float32, 'shape': []},
    'f_data': {'dtype':  tf.float32, 'shape': [None, 3]},
    'q_data': {'dtype':  tf.float32, 'shape': [None]},
    'e_weight': {'dtype': tf.float32, 'shape': []},
    'f_weights' : {'dtype':  tf.float32, 'shape': [None, 3]},
    }


@list_loader(format_dict=runner_format)
def _frame_loader(frame):
    fname, pos = frame
    bohr2ang = 0.5291772109
    with open(fname) as f:
        f.seek(pos)
        elems = []
        coord = []
        q_data = []
        f_data = []
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
            elif line == "end" :
                break
            line = f.readline().strip()
        elems = np.array(elems, np.int32)
        coord = np.array(coord, np.float32)
        f_data = np.array(f_data, np.float32)
        f_weights = np.array(f_weights, np.float32)
        cell = np.array(cell)
        q_data = np.array(q_data, np.float32)        
        data = {
            'elems': elems,
            'coord': coord,
            'e_data': e_data,
            'f_data': f_data,
            'q_data': q_data,
            'e_weight': e_weight,
            'f_weights': f_weights,
            'cell': cell
        }
        return data

def _gen_frame_list(fname):
    i = 0
    frame_list = []
    with open(fname) as f:
        for l in f:
            if re.match('.*begin', l):
                frame_list.append((fname, i))
            i += len(l)
    return frame_list

def load_runner(flist, **kwargs):
    """
    Loads runner formatted trajectory
    
    Args: 
        flist (str): one or a list of runner formatted trajectory(s)
        **kwargs: split options, see ``pinn.io.base.split_list``
    """
    if isinstance(flist, str):
        flist=[flist]
    frame_list = []
    for fname in flist:
        frame_list += _gen_frame_list(fname)
    return _frame_loader(frame_list, **kwargs)

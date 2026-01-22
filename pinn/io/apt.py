# -*- coding: utf-8 -*-
"""This file implements a loader for the APT dataset (format: https://github.com/pschienbein/AtomicPolarTensor/blob/master/examples/liquid-water/05/train.xyz)
"""

import re
from pinn.io import list_loader
import tensorflow as tf
import numpy as np

ds_spec = {
    'elems': {'dtype':  tf.int32,   'shape': [None]},
    'coord': {'dtype':  tf.float32, 'shape': [None, 3]},
    'apt': {'dtype': tf.float32, 'shape': [None, 3, 3]},
    'd_data': {'dtype':tf.float32, 'shape': [3]},
    'cell': {'dtype':tf.float32, 'shape': [3,3]},
    'oxidation': {'dtype':  tf.float32, 'shape': [None]} }

ox_dict = {'Na': 1, 'Cl':-1, 'O':-2, 'H':1}

def get_frame_list(fname):
    import mmap, re
    r = re.compile(rb'\n\d')
    with open(fname) as f:
        m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        indices = [match.span()[0]+1 for match in r.finditer(m)]
    return [0]+indices

def get_dipole_list(nfr):
    lst = []
    for i in range(nfr):
        lst.append(i)
    return lst

# If box is None the box size of the APT dataset is used
def load_apt(filename, dipolefile=None, box=None, **kwargs):
    @list_loader(ds_spec=ds_spec)
    def frame_loader(frame):
        from ase.data import atomic_numbers
        dipole = np.zeros([3])
        if dipolefile != None:
            frame,dframe = frame
            with open(dipolefile,"r") as df:
                lines=df.readlines()
                arr=lines[dframe].split()
                dipole[0],dipole[1],dipole[2] = float(arr[0]),float(arr[1]),float(arr[2])
        f = open(filename, "r")
        f.seek(frame)
        natoms = int(f.readline())
        coordlist = np.zeros([natoms,3])
        apt = np.zeros([natoms,3,3])
        elems = np.zeros([natoms])
        ox = np.zeros([natoms])
        if box is None:
            cell = np.array([[15.667,0,0],[0,15.667,0],[0,0,15.667]])
        else: 
            cell = np.array([[box[0],0,0],[0,box[1],0],[0,0,box[2]]])
        line = f.readline()
        for i in range(natoms):
            line = f.readline()
            arr = line.split()
            coordlist[i,:] = float(arr[1]),float(arr[2]),float(arr[3])
            elems[i] = atomic_numbers[arr[0]] 
            ox[i] = ox_dict[arr[0]]
            ind = 4
            for j in range(3):
                for k in range(3):
                    apt[i,j,k] = float(arr[ind])
                    ind+=1
        return {'elems': elems, 'coord': coordlist, 'apt': apt, 'd_data': dipole, 'cell': cell, 'oxidation': ox}
    frames = get_frame_list(filename)
    if dipolefile != None:
        dipoleframes = get_dipole_list(len(frames))
        frames = list(zip(frames, dipoleframes))
    return frame_loader(frames, **kwargs)

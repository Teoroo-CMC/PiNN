# -*- coding: utf-8 -*-
"""This file implements loaders for the QM7b dataset.
"""

import os, re, warnings
from pinn.io import sparse_batch
from pinn.io import list_loader
import tensorflow as tf
import numpy as np

ds_spec = {
    'elems': {'dtype':  tf.int32,   'shape': [None]},
    'coord': {'dtype':  tf.float32, 'shape': [None, 3]},
    'ptensor': {'dtype': tf.float32, 'shape': [3, 3]}}


def get_frame_list(fname):
    import mmap, re
    r = re.compile(b'\n\d')
    with open(fname) as f:
        m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        indices = [match.span()[0]+1 for match in r.finditer(m)]
    return [0]+indices


def load_qm7b(filename, **kwargs):
    @list_loader(ds_spec=ds_spec)
    def frame_loader(frame):
        from math import pi,sqrt
        from ase.data import atomic_numbers
        f = open(filename, "r")
        f.seek(frame)
        natoms = int(f.readline())
        coordlist = np.zeros([natoms,3])
        parray = np.zeros([6])
        elems = np.zeros([natoms])
        pline = f.readline()
        pstring = ''
        j = 0
        # search for the pol string, use the lazy '.+' to strip the possible space
        for char in re.search(r'b3lyp_pol="\s*(.+?)\s*"', pline).group(1):
            if (char == ' '):
                parray[j] = float(pstring)
                j += 1
                pstring = ''
            else: 
                pstring += char
        parray[5] = float(pstring)
        pmatrix = np.diagflat(parray[:3])
        pmatrix[0,1] = parray[3]
        pmatrix[1,0] = pmatrix[0,1]
        pmatrix[0,2] = parray[4]
        pmatrix[2,0] = pmatrix[0,2]
        pmatrix[1,2] = parray[5]
        pmatrix[2,1] = pmatrix[1,2]
        for i in range(natoms):
            xyz = f.readline()
            element, x, y, z = xyz.split()
            coordlist[i,:] = float(x),float(y),float(z)
            elems[i] = atomic_numbers[element] 
        return {'elems': elems, 'coord': coordlist, 'ptensor': pmatrix}
    frames = get_frame_list(filename)
    return frame_loader(frames, **kwargs)

ds_spec_egap = {
    'elems': {'dtype':  tf.int32,   'shape': [None]},
    'coord': {'dtype':  tf.float32, 'shape': [None, 3]},
    'ptensor': {'dtype': tf.float32, 'shape': [3, 3]},
    'egap': {'dtype': tf.float32, 'shape': []}}

@list_loader(ds_spec=ds_spec_egap)
def load_qm7b_file(fname):
    from ase.data import atomic_numbers
    f = open(fname,'r')
    data = f.readlines()
    f.close()
    props = data[1].split(',')
    HOMO, LUMO = float(props[-2]),float(props[-1])
    egap = tf.constant(LUMO-HOMO,dtype=tf.float32)
    pmatrix = np.zeros([3,3])
    pmatrix[0,0],pmatrix[1,1],pmatrix[2,2] = props[3],props[4],props[5]
    pmatrix[0,1],pmatrix[0,2],pmatrix[1,2] = props[6],props[7],props[8]
    pmatrix[1,0],pmatrix[2,0],pmatrix[2,1] = pmatrix[0,1],pmatrix[0,2],pmatrix[1,2]
    natoms = int(data[0])
    coord = np.zeros([natoms,3])
    elems = np.zeros([natoms])
    for i in range(natoms):
       datstr = ' '.join(data[2+i].split())
       element,x,y,z = datstr.split(' ')
       elems[i],coord[i,0],coord[i,1],coord[i,2] = atomic_numbers[element],float(x),float(y),float(z)
    return {'elems':elems,'coord':coord,'ptensor': pmatrix,'egap': egap}
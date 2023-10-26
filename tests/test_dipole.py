# -*- coding: utf-8 -*-
import numpy as np

def _get_dipole_data():
    from ase.build import molecule

    water = molecule('H2O')

    q_O = -0.8476
    q_H = abs(q_O)/2
    q = np.array([[q_O, q_H, q_H]])

    coord, elems, d_data = [], [], []
    for i in range(120):
        water.rotate(3, 'x')
        r = water.positions
        dipole = q @ r

        coord.append(water.positions)
        elems.append(water.numbers)
        d_data.append(dipole.flatten())

    data = {
            'coord': np.array(coord),
            'elems': np.array(elems),
            'd_data': np.array(d_data)
            }

    return data

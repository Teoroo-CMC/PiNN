# -*- coding: utf-8 -*-
import re
import numpy as np
import tensorflow as tf
from ase.data import atomic_numbers
from pinn.io import list_loader


def _gen_frame_list(fname):
    import mmap
    i = 0
    frame_list = []
    steps = []
    f = open(fname, 'r')
    m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    for match in re.finditer(b' i =\s*(\d+),', m):
        frame_list.append((fname, match.span()[0]))
        steps.append(int(match.group(1)))
    return steps, frame_list


@list_loader(pbc=True, force=True)
def _frame_loader(frame):
    def _read(frame):
        elems = []
        coord = []
        f = open(frame[0], 'r')
        f.seek(frame[1])
        f.readline()
        while True:
            line = f.readline().split()
            if len(line) <= 1:
                break
            elems.append(atomic_numbers[line[0]])
            coord.append(line[1:4])
        return elems, coord
    pos_frame, frc_frame, e_data, cell = frame
    elems, coord = _read(pos_frame)
    _, f_data = _read(frc_frame)
    data = {'coord': coord, 'cell': cell, 'elems': elems,
            'e_data': e_data, 'f_data': np.array(f_data, np.float32)}
    return data


def load_cp2k(coord_file, force_file, ener_file, cell_file, **kwargs):
    """Loads CP2K formatted trajectories

    CP2K outputs the coord, force, energy and cell in separate files.
    It is assumed that different files come in consistent units (no
    unit conversion is done in the loader).

    Args:
        coord_file: one or a list of CP2K .xyz files for coordinates
        force_file: one or a list of CP2K .xyz files for forces
        ener_file: one or a list of CP2K .ener files
        cell_file: one or a list of CP2K .cell files
        **kwargs: split options, see ``pinn.io.base.split_list``

    """

    if isinstance(coord_file, str):
        flist = [(coord_file, force_file, ener_file, cell_file)]
    else:
        flist = zip(coord_file, force_file, ener_file, cell_file)

    frame_list = []
    for fname in flist:
        # coord and force files are large, we just skim through them
        # to get a list of locations where frame starts, the energy
        # and cell we can just load into memory.
        coord_file, force_file, ener_file, cell_file = fname
        step_c, frame_c = _gen_frame_list(coord_file)
        step_f, frame_f = _gen_frame_list(force_file)
        ener = np.loadtxt(ener_file)[:, 4]
        cell = np.loadtxt(cell_file)[:, 2:-1]
        cell = cell.reshape(cell.shape[0], 3, 3)
        # do some minimal size checks
        assert step_c == step_f,\
            "Mismatching {} and {}".format(coord_file, force_file)
        assert cell.shape[0] == len(step_c),\
            "Mismatching {} and {}".format(coord_file, cell_file)
        assert ener.shape[0] == len(step_c),\
            "Mismatching {} and {}".format(coord_file, ener_file)
        # update the frame list
        frame_list += list(zip(frame_c, frame_f, ener, cell))

    return _frame_loader(frame_list, **kwargs)

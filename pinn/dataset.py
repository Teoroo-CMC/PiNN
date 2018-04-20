import numpy as np
import tensorflow as tf
import random, h5py
from ase import Atoms

class from_ani():
    def __init__(self, files, n_training=50000, seed=None, n_atoms=None):
        data_list = []
        for file in files:
            store = h5py.File(file)
            for g in store.keys():
                group = store[g]
                for k in group.keys():
                    dat = (file, '/{}/{}'.format(g,k))
                    data_list.append(dat)
        if seed is not None:
            random.seed(seed)
        random.shuffle(data_list)

        self.data_list = data_list
        self.training = self.data_list[0:n_training]
        self.testing = self.data_list[n_training:]

        if n_atoms is None:
            n_atoms = self.get_max_natoms()
        self.n_atoms = n_atoms

    def get_max_natoms(self):
        n_atoms = 0
        for file, path in self.data_list:
            n_atoms = max(n_atoms, h5py.File(file)[path]['coordinatesHE'].shape[1])
        return n_atoms

    def get_training(self, p_filter, dtypes):
        g = ani_generator(self.training, p_filter)
        dataset = tf.data.Dataset.from_generator(g, dtypes)
        return dataset

    def get_testing(self, p_filter, dtypes):
        g = ani_generator(self.test, p_filter)
        dataset = tf.data.Dataset.from_generator(g, dtypes)
        return dataset

class ani_generator():
    def __init__(self, data_list, p_filter):
        self.data_list = data_list
        self.p_filter = p_filter

    def __call__(self):
        for file,path in self.data_list:
            data = h5py.File(file)[path]
            c = data['coordinates'].value
            e = data['energies'].value
            p = data['species'].value
            p = self.p_filter.parse(Atoms([a.decode('ascii') for a in p]))
            for i in range(e.shape[0]):
                yield {'c_in':c[i], 'p_in':p, 'e_in':[e[i]]}

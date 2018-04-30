import numpy as np
import tensorflow as tf
import random, h5py
from ase import Atoms
import os


class from_ase_traj():
    def __init__(self, traj, n_atoms=29):
        self.traj = traj
        self.n_atoms = n_atoms

    def get_training(self, dtypes):
        generator = lambda : traj_generater(self.traj)
        dataset = tf.data.Dataset.from_generator(generator, dtypes)
        return dataset

def traj_generater(traj):
    for atoms in traj:
        c_mat = atoms.get_positions()
        a_mat = atoms.get_atomic_numbers()
        e_mat = [atoms.get_potential_energy()]
        yield {'c_in': c_mat, 'a_in': a_mat, 'e_in': e_mat}


class from_tfrecord_ani():
    def __init__(self, data_path, n_atoms=26):
        self.train_file = os.path.join(data_path, 'train.tfrecord')
        self.test_file = os.path.join(data_path, 'test.tfrecord')
        self.vali_file = os.path.join(data_path, 'vali.tfrecord')
        self.n_atoms = n_atoms

    def get_training(self, dtypes):
        return _tfrecord_to_dataset(self.train_file, dtypes)


def _tfrecord_to_dataset(record_file, dtypes):
    def _record_to_dataset(tfrecord, dtypes):
        feature_dtypes = {
            'atoms_raw': tf.FixedLenFeature([], tf.string),
            'energ_raw': tf.FixedLenFeature([], tf.string),
            'coord_raw': tf.FixedLenFeature([], tf.string),
            'n_samples': tf.FixedLenFeature([], tf.int64),
            'n_atoms': tf.FixedLenFeature([], tf.int64),
        }
        record = tf.parse_single_example(tfrecord, features=feature_dtypes)
        # HARD-CODED datatypes
        atoms = tf.decode_raw(record['atoms_raw'], tf.int32)
        energ = tf.decode_raw(record['energ_raw'], tf.float64)
        coord = tf.decode_raw(record['coord_raw'], tf.float32)

        n_samples = tf.cast(record['n_samples'], tf.int64)
        n_atoms = tf.cast(record['n_atoms'], tf.int64)

        coord_shape = tf.stack([n_samples, n_atoms, 3])
        coord = tf.reshape(coord, coord_shape)
        energ = tf.expand_dims(tf.cast(energ, tf.float32),1)

        dataset_c = tf.data.Dataset.from_tensor_slices(coord)
        dataset_e = tf.data.Dataset.from_tensor_slices(energ)
        dataset_a = tf.data.Dataset.from_tensors(atoms).repeat(n_samples)
        dataset = tf.data.Dataset.zip({'c_in': dataset_c,
                                       'a_in': dataset_a,
                                       'e_in': dataset_e,})
        return dataset

    dataset = tf.data.TFRecordDataset(record_file)
    dataset = dataset.flat_map(lambda x: _record_to_dataset(x, dtypes))
    return dataset



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
                yield {'c_in':c[i], 'a_in':p, 'e_in':[e[i]]}

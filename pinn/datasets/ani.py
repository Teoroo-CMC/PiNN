# -*- coding: utf-8 -*-
"""H5 dataset that follows the format of ANI1 dataset

ANI-1, A data set of 20 million calculated off-equilibrium conformations
for organic molecules.
Justin S. Smith, Olexandr Isayev, Adrian E. Roitberg.
DOI: 10.1038/sdata.2017.193

The dataset is split on a 'molecular' basis, meaninig the same molecule
show up in only one of the train/vali/test datasets.

Todo:
    Rewrite the set
"""

class ANI_H5_dataset():

    def __init__(self, files, n_atoms=None,
                 split_ratio=[0.9, 0.05, 0.05], shuffle=True, seed=None):
        """
        Args:
            files (list): A list of database files or one database file
            natoms (:obj:`int`, optional): The maximum number of atoms in the
                dataset
        """
        import h5py
        import random

        if type(files) == str:
            files = [files]

        data_list = []
        self.n_atoms = 0

        for fname in files:
            store = h5py.File(fname)
            for g in store.keys():
                group = store[g]
                for k in group.keys():
                    path = '/{}/{}'.format(g, k)
                    dat = (fname, path)
                    data_list.append(dat)
                    n_atoms = store[path]['coordinates'].shape[1]
                    self.n_atoms = max(self.n_atoms, n_atoms)

        random.seed(seed)
        random.shuffle(data_list)
        n_train = int(len(data_list) * split_ratio[0])
        n_test = int(len(data_list) * sum(split_ratio[:2]))

        self._train_list = data_list[:n_train]
        self._test_list = data_list[n_train:n_test]
        self._vali_list = data_list[n_test:]

    def get_train(self, shuffle=10000, batch_size=100, dtype=tf.float32):
        dataset = _ani_to_dataset(self._train_list, dtype)
        padded_shapes = {
            'coord': [self.n_atoms, 3],
            'atoms': [self.n_atoms],
            'e_data': []
        }
        dataset = dataset.shuffle(shuffle).repeat()
        dataset = dataset.padded_batch(batch_size, padded_shapes,
                                       drop_remainder=True)
        dataset = dataset.prefetch(30000)
        return dataset

    def get_test(self, dtype=tf.float32):
        dataset = _ani_to_dataset(self._test_list, dtype)

        return dataset

    def get_vali(self, shuffle=10000, batch_size=10, dtype=tf.float32):
        dataset = _ani_to_dataset(self._vali_list, dtype)
        padded_shapes = {
            'coord': [self.n_atoms, 3],
            'atoms': [self.n_atoms],
            'e_data': []
        }
        dataset = dataset.shuffle(shuffle).repeat()
        dataset = dataset.padded_batch(batch_size, padded_shapes,
                                       drop_remainder=True)
        return dataset


def _ani_generator(datalist):
    import h5py
    import ase

    for fname, path in datalist:
        data = h5py.File(fname)[path]

        atoms = [ase.data.atomic_numbers[i.decode('ascii')]
                 for i in data['species'].value]
        coord = data['coordinates'].value
        energ = data['energies'].value

        size = energ.shape[0]

        for i in range(size):
            data = {'coord': coord[i],
                    'atoms': atoms,
                    'e_data': energ[i]}
            yield data


def _ani_to_dataset(datalist, dtype):
    dataset = tf.data.Dataset.from_generator(
        lambda: _ani_generator(datalist),
        {'coord': dtype, 'atoms': tf.int32, 'e_data': dtype},
        {'coord': [None, 3], 'atoms': [None], 'e_data': []},
    )
    return dataset

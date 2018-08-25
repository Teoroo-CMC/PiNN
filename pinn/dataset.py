"""
    Datasets in PiNN feeds the atomic data to models
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Output shapes:
    'coord':   [natoms, 3]
    'atoms':   [natoms]
    'energy':  []

"""

import os
import tensorflow as tf


class QSO_XML_dataset():
    """XML format for first-principle simulations recommanded here
    http://www.quantum-simulation.org/

    The dataset on a image basis, meaning each xml file is splited to
    three portions
    """

    def __init__(self, files, n_atoms=None, pbc=1,
                 split_ratio=[0.9, 0.05, 0.05], shuffle=True, seed=None):
        """
        Args:
            files (list): A list of xml files or one xml file
            natoms (:obj:`int`, optional): The maximum number of atoms in the
                 dataset
            pbc (int): Specify the pbc type, 1 for orthoganal cell
        """
        from lxml import etree
        import random

        self.n_atoms = n_atoms
        self.pbc = pbc

        random.seed(seed)
        random.shuffle(files)
        self._train_list = []
        self._test_list = []
        self._vali_list = []
        self.n_atoms = 0

        for fname in files:
            tree = etree.parse(fname)
            size = int(tree.findall('iteration')[-1].values()[0])
            n_atoms = len(tree.find('iteration').find('atomset').findall('atom'))
            self.n_atoms = max(n_atoms, self.n_atoms)

            split_list = [i+1 for i in range(size)]
            random.shuffle(split_list)
            n_train = int(len(split_list) * split_ratio[0])
            n_test = int(len(split_list) * sum(split_ratio[:2]))

            self._train_list.append((fname, split_list[: n_train]))
            self._test_list.append((fname, split_list[n_train: n_test]))
            self._vali_list.append((fname, split_list[n_test:]))

    def get_train(self, shuffle=1000, batch_size=100, dtype=tf.float32):
        dataset = _qso_xml_to_dataset(self._train_list, self.pbc, dtype)

        padded_shapes = {
            'coord': [self.n_atoms, 3],
            'atoms': [self.n_atoms],
            'e_data': []}
        if self.pbc == 1:
            padded_shapes['cell'] = [3]

        dataset = dataset.shuffle(shuffle).repeat()
        if batch_size:
            dataset = dataset.padded_batch(batch_size, padded_shapes)

        return dataset

    def get_test(self, dtype=tf.float32):
        dataset = _qso_xml_to_dataset(self._vali_list, self.pbc, dtype)
        return dataset

    def get_vali(self, dtype=tf.float32):
        dataset = _qso_xml_to_dataset(self._vali_list, self.pbc, dtype)
        return dataset


def _qso_xml_generator(filelist, pbc):
    from lxml import etree
    import ase
    # the QSO xml format uses Bohr as the length unit
    scale = ase.units.Bohr / ase.units.Angstrom
    for fname, iters in filelist:
        species = {}
        for event, elem in etree.iterparse(fname):
            if elem.tag == "species":
                name = elem.values()[0]
                atomic_number = elem.find('atomic_number')
                species[name] = int(atomic_number.text)
            if elem.tag == "iteration" and int(elem.values()[0]) in iters:
                atomset = elem.find('atomset')
                atoms = atomset.findall('atom')
                coord = [[float(i) * scale for i in
                          atom.find('position').text.split()]
                         for atom in atoms]
                atoms = [species[atom.values()[1]] for atom in atoms]
                energy = float(elem.find('etotal').text)
                data = {
                    'coord': coord,
                    'atoms': atoms,
                    'e_data': energy
                }
                if pbc == 1:
                    cell = atomset.find('unit_cell')
                    cell = [float(cell.values()[i].split()[i]) * scale
                            for i in range(3)]
                    data['cell'] = cell
                yield data


def _qso_xml_to_dataset(filelist, pbc, dtype):
    dtypes = {'coord': dtype, 'atoms': tf.int32, 'e_data': dtype}
    shapes = {'coord': [None, 3], 'atoms': [None], 'e_data': []}
    if pbc == 1:
        dtypes['cell'] = dtype
        shapes['cell'] = [3]
    elif pbc == 2:
        dtypes['cell'] = dtype
        shapes['cell'] = [3, 3]
    dataset = tf.data.Dataset.from_generator(
        lambda: _qso_xml_generator(filelist, pbc), dtypes, shapes)

    return dataset


class ANI_H5_dataset():
    """H5 dataset that follows the format of ANI1 dataset

    ANI-1, A data set of 20 million calculated off-equilibrium conformations
    for organic molecules.
    Justin S. Smith, Olexandr Isayev, Adrian E. Roitberg.
    DOI: 10.1038/sdata.2017.193

    The dataset is split on a 'molecular' basis, meaninig the same molecule
    show up in only one of the train/vali/test datasets.
    """

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


class tfrecord_dataset():
    """Dataset stored with the tfrecord format

    Dataset is splited into 'chunks' of data, and
    """

    def __init__(self, folder, n_atoms=None):
        sef.folder = folder

    def get_train(self, dtype):
        fname = os.path.join(self.folder, 'train.tfrecord')
        return _tfrecord_to_dataset(fname, dtype)


def _tfrecord_to_dataset(fname):
    dataset = tf.data.TFRecordDataset(fname)
    raw_dtypes = {
        'atoms': tf.int32,
        'energ': tf.float32,
        'coord': tf.float32}
    raw_shapes = {
        'atoms': lambda size, natoms: [1, natoms],
        'energ': lambda size, natoms: [size, 1],
        'coord': lambda size, natoms: [size, natoms, 3]
    }
    out_dtypes = {
        'atoms': tf.int32,
        'energ': dtype,
        'coord': dtype}
    dataset = dataset.flat_map(
        lambda data: _record_to_structrue(
            data, raw_dtypes, raw_shapes, out_dtypes))
    return dataset


def _record_to_structure(record):
    """Convert one record (chunk of data) to a dataset

    The record should be arranged as following:
        size          :(int64)  number of structures in this record
        natoms        :(int64)  maximum number of atoms in each structure
        {featrue}_raw :(string) raw data of the features
    """
    feature_dtypes = {
        'size':  tf.FixedLenFeature([], tf.int64),
        'natoms': tf.FixedLenFeature([], tf.int64),
        'atoms_raw': tf.FixedLenFeature([], tf.string),
        'energ_raw': tf.FixedLenFeature([], tf.string),
        'coord_raw': tf.FixedLenFeature([], tf.string)}

    datasets = {}
    for key in raw_shapes:
        data = tf.decode_raw(record['{}_raw'.format(key)], raw_dtypes[key])
        data = tf.reshape(data, raw_shape[key](size, natoms))
        data = tf.cast(data, out_dtypes[key])
        dataset = tf.data.Dataset.from_tensor_slices(data)
        if data.shape[0] == 1:
            dataset = dataset.repeat(size)
        datasets[key] = data
    dataset = tf.data.Dataset.zip(datasets)

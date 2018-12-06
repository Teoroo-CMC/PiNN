# -*- coding: utf-8 -*-
"""Datasets from quantum/simulation.org

URL: http://www.quantum-simulation.org/

Todo:
    rewrte with datalist_dataset and document
"""

class QSO_XML_dataset():
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

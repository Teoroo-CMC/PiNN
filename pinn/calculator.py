# -*- coding: utf-8 -*-
"""ASE calcualtor for to use with PiNN
"""

import tensorflow as tf
from ase.calculators.calculator import Calculator


class PiNN_calc(Calculator):
    def __init__(self, model=None, atoms=None):
        """
        Args:
            model: tf.Estimator object
        """
        Calculator.__init__(self)
        self.implemented_properties = ['energy', 'forces', 'stress']
        self.model = model
        self.size = 0
        self.atoms = atoms
        self.predictor = None

    def _generator(self):
        while True:
            coord = self._atoms_to_calc.get_positions()
            atoms = self._atoms_to_calc.get_atomic_numbers()
            data = {'coord': coord, 'atoms': atoms}
            if self._atoms_to_calc.pbc.any():
                cell = self._atoms_to_calc.cell
                data['cell'] = cell
            yield data

    def get_predictor(self, dtype=tf.float32):
        if self.predictor is not None:
            return self.predictor

        self.size = len(self._atoms_to_calc)

        dtypes = {'coord': dtype, 'atoms': tf.int32}
        shapes = {'coord': [self.size, 3], 'atoms': [self.size]}
        properties = ['energy', 'forces']

        if self._atoms_to_calc.pbc.any():
            shapes['cell'] = self._atoms_to_calc.cell.shape
            dtypes['cell'] = dtype

        self.predictor = self.model.predict(
            input_fn=lambda: tf.data.Dataset.from_generator(
                self._generator, dtypes, shapes).batch(1, drop_remainder=True),
            predict_keys=properties)
        return self.predictor

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=None):
        if atoms is None:
            self._atoms_to_calc = self.atoms
        else:
            self._atoms_to_calc = atoms

        if len(self._atoms_to_calc) != self.size:
            print('Generating the graph')
            self.predictor = None

        predictor = self.get_predictor()
        self.results = next(predictor)

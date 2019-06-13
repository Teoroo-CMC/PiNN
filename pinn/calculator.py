# -*- coding: utf-8 -*-
"""ASE calcualtor for to use with PiNN
"""
import numpy as np
import tensorflow as tf
from ase.calculators.calculator import Calculator


class PiNN_calc(Calculator):
    def __init__(self, model=None, atoms=None, unit=1):
        """
        Args:
            model: tf.Estimator object
        """
        Calculator.__init__(self)
        self.implemented_properties = ['energy', 'forces', 'stress']
        self.model = model
        self.pbc = False
        self.unit = unit
        self.atoms = atoms
        self.predictor = None

    def _generator(self):
        while True:
            if self._atoms_to_calc.pbc.any():
                data = {
                    'cell': self._atoms_to_calc.cell[np.newaxis,:,:],
                    'coord': self._atoms_to_calc.get_positions(wrap=True),
                    'ind_1': np.zeros([len(self._atoms_to_calc), 1]),
                    'elems': self._atoms_to_calc.numbers}
            else:
                data = {
                    'coord': self._atoms_to_calc.positions,
                    'ind_1': np.zeros([len(self._atoms_to_calc), 1]),
                    'elems': self._atoms_to_calc.numbers}
            yield data

    def get_predictor(self, dtype=tf.float32):
        if self.predictor is not None:
            return self.predictor

        self.size = len(self._atoms_to_calc)

        dtypes = {'coord': dtype, 'elems': tf.int32, 'ind_1': tf.int32}
        shapes = {'coord': [None, 3], 'elems': [None], 'ind_1': [None, 1]}
        properties = ['energy', 'forces', 'stress']

        if self._atoms_to_calc.pbc.any():
            shapes['cell'] = [1,3,3]
            dtypes['cell'] = dtype
            self.pbc=True
        else:
            self.pbc=False

        self.predictor = self.model.predict(
            input_fn=lambda: tf.data.Dataset.from_generator(
                self._generator, dtypes, shapes),
            predict_keys=properties)
        return self.predictor

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=None):
        if atoms is None:
            self._atoms_to_calc = self.atoms
        else:
            self._atoms_to_calc = atoms
            
        if self._atoms_to_calc.pbc.any() != self.pbc:
            print('PBC condition changed, reset the predictor.')
            self.predictor = None
            
        predictor = self.get_predictor()
        results = next(predictor)
        results = {k: v*self.unit for k,v in results.items()}
        
        if 'stress' in properties:
            results['stress'] /= self._atoms_to_calc.get_volume()
            results['stress'] = results['stress'].flat[[0, 4, 8, 5, 2, 1]]
        self.results = results


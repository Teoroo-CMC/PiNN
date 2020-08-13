# -*- coding: utf-8 -*-
"""ASE calcualtor for to use with PiNN"""

import numpy as np
import tensorflow as tf
from ase.calculators.calculator import Calculator

def get(model_spec, **kwargs):
    """Get a calculator from a trained model.

    The positional argument will be passed to `pinn.get_model`, keyword
    arguments will be passed to the calculator.
    """
    from pinn import get_model
    return  PiNN_calc(get_model(model_spec), **kwargs)

class PiNN_calc(Calculator):
    def __init__(self, model=None, atoms=None, to_eV=1.0,
                 properties=['energy', 'forces', 'stress']):
        """PiNN interface with ASE as a calculator

        Args:
            model: tf.Estimator object
            atoms: optional, ase Atoms object
            properties: properties to calculate.
                the properties to calculate is fixed for each calculator,
                to avoid resetting the predictor during get_* calls.
        """
        Calculator.__init__(self)
        self.implemented_properties = properties
        self.model = model
        self.pbc = False
        self.atoms = atoms
        self.predictor = None
        self.to_eV = to_eV

    def _generator(self):
        while True:
            if self._atoms_to_calc.pbc.any():
                data = {
                    'cell': self._atoms_to_calc.cell[np.newaxis, :, :],
                    'coord': self._atoms_to_calc.positions,
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
        properties = self.implemented_properties

        if self._atoms_to_calc.pbc.any():
            shapes['cell'] = [1, 3, 3]
            dtypes['cell'] = dtype
            self.pbc = True
        else:
            self.pbc = False

        self.predictor = self.model.predict(
            input_fn=lambda: tf.data.Dataset.from_generator(
                self._generator, dtypes, shapes),
            predict_keys=properties)
        return self.predictor

    def calculate(self, atoms=None,
                  properties=None, system_changes=None):
        """Run a calculation. 

        The properties and system_changes are ignored here since we do
        not want to reset the predictor frequently. Whenever
        calculator is executed, the predictor is run. The calculate
        method will not be executed if atoms are not changed since
        last run (this should be haneled by
        ase.calculator.Calculator).
        """
        if atoms is not None:
            self.atoms = atoms.copy()
        self._atoms_to_calc = self.atoms

        if self._atoms_to_calc.pbc.any() != self.pbc and self.predictor:
            print('PBC condition changed, reset the predictor.')
            self.predictor = None

        predictor = self.get_predictor()
        results = next(predictor)
        # the below conversion works for energy, forces, and stress,
        # it is assumed that the distance unit is angstrom
        results = {k: v*self.to_eV
                   if k in ['energy', 'forces', 'stress'] else v
                   for k, v in results.items()}
        if 'stress' in results and self._atoms_to_calc.pbc.all():
            results['stress'] = results['stress'].flat[[0, 4, 8, 5, 2, 1]]
        self.results = results

# -*- coding: utf-8 -*-
"""ASE calcualtor for to use with PiNN"""

import numpy as np
import tensorflow as tf
from ase.calculators.calculator import Calculator
from pinn.models.base import (
    dtype_from_params, set_infer_dtype, tf_dtype_from_name,
)


class PiNN_calc(Calculator):
    def __init__(self, model=None, atoms=None, to_eV=1.0,
                 properties=['energy', 'forces', 'stress'],
                 checkpoint_path=None, default_dtype=None):
        """PiNN interface with ASE as a calculator

        Args:
            model: tf.Estimator object
            atoms: optional, ase Atoms object
            checkpoint_path: specify the checkpoint to use
            properties: properties to calculate.
                the properties to calculate is fixed for each calculator,
                to avoid resetting the predictor during get_* calls.
            default_dtype: MACE-style calculator arg (not a YAML key).
                ``None`` follows ``settings.dtype``. If it differs from
                training, checkpoint weights are cast once at load
                (like ``model.float()`` / ``model.double()``) and the
                whole network runs in that dtype. ASE MD stays float64.
        """
        Calculator.__init__(self)
        self.implemented_properties = properties
        self.model = model
        self.pbc = False
        self.atoms = atoms
        self.predictor = None
        self.to_eV = to_eV
        self.ckpt_path = checkpoint_path
        self.default_dtype = default_dtype

    def _dtype_name(self):
        if self.default_dtype is not None:
            return self.default_dtype
        params = getattr(self.model, 'params', None)
        return dtype_from_params(params)

    def _tf_dtype(self):
        return tf_dtype_from_name(self._dtype_name())

    def _generator(self):
        while True:
            atoms = self._atoms_to_calc
            if atoms.pbc.any():
                data = {
                    'cell': np.asarray(atoms.cell)[np.newaxis, :, :],
                    'coord': atoms.positions,
                    'ind_1': np.zeros([len(atoms), 1]),
                    'elems': atoms.numbers}
            else:
                data = {
                    'coord': atoms.positions,
                    'ind_1': np.zeros([len(atoms), 1]),
                    'elems': atoms.numbers}
            yield data

    def get_predictor(self, dtype=None):
        if self.predictor is not None:
            return self.predictor

        self.size = len(self._atoms_to_calc)
        if dtype is None:
            dtype = self._tf_dtype()
        set_infer_dtype(self._dtype_name())

        dtypes = {'coord': dtype, 'elems': tf.int32, 'ind_1': tf.int32}
        shapes = {'coord': [None, 3], 'elems': [None], 'ind_1': [None, 1]}
        properties = self.implemented_properties

        if self._atoms_to_calc.pbc.any():
            shapes['cell'] = [1, 3, 3]
            dtypes['cell'] = dtype
            self.pbc = True
        else:
            self.pbc = False

        def _input_fn():
            ds = tf.data.Dataset.from_generator(
                self._generator, dtypes, shapes)
            # Pull exactly one element per predict step. TF>=2.15 otherwise
            # autotunes a prefetch that reads the generator ahead of time, so a
            # next() would return the prediction for the PREVIOUS atoms (the
            # calculator mutates self._atoms_to_calc in place between calls).
            options = tf.data.Options()
            options.autotune.enabled = False
            options.experimental_optimization.apply_default_optimizations = False
            return ds.with_options(options)

        self.predictor = self.model.predict(
            input_fn=_input_fn,
            predict_keys=properties,
            checkpoint_path=self.ckpt_path)
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

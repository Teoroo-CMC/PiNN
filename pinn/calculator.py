import tensorflow as tf
from ase.calculators.calculator import Calculator


class PiNN_calc(Calculator):
    def __init__(self, model=None, atoms=None):
        Calculator.__init__(self)
        self.implemented_properties = ['energy', 'forces']
        self.model = model
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

        dtypes = {'coord': dtype, 'atoms': tf.int32}
        shapes = {'coord': [None, 3], 'atoms': [None]}
        properties = ['energy', 'forces']

        self.predictor = self.model.predict(
            input_fn=lambda: tf.data.Dataset.from_generator(
                self._generator, dtypes, shapes).batch(1),
            predict_keys=properties)
        return self.predictor

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=None):
        if atoms is None:
            self._atoms_to_calc = self.atoms
        else:
            self._atoms_to_calc = atoms

        predictor = self.get_predictor()
        self.results = next(predictor)

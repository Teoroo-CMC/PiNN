import tensorflow as tf
import numpy as np
from ase.calculators.calculator import Calculator

import pinn.filters as filters
import pinn.layers as layers
from pinn.model import pinn_model


class PINN(Calculator):
    def __init__(self, model=None):
        Calculator.__init__(self)
        self.implemented_properties = ['energy', 'forces']
        if type(model) == str:
            self.model = pinn_model()
            self.model.load(model)
        else:
            self.model = model
        self.sess = None
        self.tensors = None


    def get_tensors(self):
        if self.tensors is None:
            tf.reset_default_graph()
            n_atoms = len(self.atoms)
            c_in = tf.placeholder(self.model.dtype, shape=(1, n_atoms, 3))
            c_flat = tf.reshape(c_in, [n_atoms*3])
            c_in_2 = tf.reshape(c_flat, [1, n_atoms, 3])
            tensors = self.model.get_tensors(c_in_2)
            tensors['c_in'] = c_in
            self.tensors = tensors
        return self.tensors

    def get_sess(self, config=None):
        if self.sess is None:
            self.sess = tf.Session(config=config)
        return self.sess

    def calculate(self, atoms=None, properties=['energy'],
                  system_chages=None):
        self.atoms = atoms
        tensors = self.get_tensors()
        c_in = tensors['c_in']
        p_in = tensors['p_in']
        # Properties to calculate
        if 'forces' in properties and 'forces' not in self.tensors:
            tensors['forces'] = tf.gradients(tensors['energy'],
                                             tensors['c_in'])[0][0]
        if 'hessian' in properties and 'hessian' not in self.tensors:
            tensors['hessian'] = tf.hessians(tensors['energy'],
                                             tensors['c_flat'])[0]
        to_calc = {}
        for item in properties:
            to_calc[item] = self.tensors[item]

        # Run the calculation
        c_mat = [atoms.get_positions()]
        p_mat = [self.model.p_filter.parse(atoms)]
        sess = self.get_sess()
        vars = self.get_tensors()
        sess.run(tf.global_variables_initializer())
        self.results = sess.run(
            to_calc, feed_dict={c_in: c_mat, p_in: p_mat})

    def get_vibrational_modes(self, atoms=None):
        if atoms is None:
            atoms = self.atoms
        self.calculate(atoms, properties=['hessian'])
        hessian = self.results['hessian']
        freqs, modes = np.linalg.eig(hessian)
        modes = [modes[:,i].reshape(len(atoms),3)
                 for i in range(len(freqs))]
        return freqs, modes

# import tensorflow as tf
# import numpy as np
# from ase.calculators.calculator import Calculator
# from pinn.model import pinn_model


# class PINN(Calculator):
#     def __init__(self, model=None, sess_config=None):
#         Calculator.__init__(self)
#         self.implemented_properties = ['energy', 'forces']
#         if type(model) == str:
#             self.model = pinn_model()
#             self.model.load(model)
#         else:
#             self.model = model
#         self.sess_config = sess_config
#         self.tensors = None
#         self.sess = None

#     def get_tensors(self):
#         if self.tensors is None:
#             n_atoms = len(self.atoms)
#             c_in = tf.placeholder(self.model.dtype, shape=(1, n_atoms, 3))
#             a_in = tf.placeholder(
#                 tf.int32,
#                 shape=(1,n_atoms))
#             data = {'c_in': c_in, 'a_in': a_in}
#             data = self.model.p_filter.parse(data, self.model.dtype)
#             inputs = self.model.get_inputs(data)
#             energy = self.model.get_energy(inputs)
#             data['energy'] = energy
#             self.tensors = data
#         return self.tensors

#     def get_sess(self):
#         if self.sess is None:
#             self.sess = tf.Session(config=self.sess_config)
#         return self.sess

#     def calculate(self, atoms=None, properties=['energy'],
#                   system_chages=None):
#         self.atoms = atoms
#         tensors = self.get_tensors()
#         c_in = tensors['c_in']
#         a_in = tensors['a_in']
#         # Properties to calculate
#         if 'forces' in properties and 'forces' not in self.tensors:
#             tensors['forces'] = - tf.gradients(tensors['energy'],
#                                              tensors['c_in'])[0][0]
#         if 'hessian' in properties and 'hessian' not in self.tensors:
#             tensors['hessian'] = tf.hessians(tensors['energy'],
#                                              tensors['c_flat'])[0]
#         to_calc = {}
#         for item in properties:
#             to_calc[item] = self.tensors[item]

#         # Run the calculation
#         c_mat = [atoms.get_positions()]
#         a_mat = [atoms.get_atomic_numbers()]

#         sess = self.get_sess()
#         vars = self.get_tensors()
#         sess.run(tf.global_variables_initializer())
#         self.results = sess.run(
#             to_calc, feed_dict={c_in: c_mat, a_in: a_mat})

#     def get_vibrational_modes(self, atoms=None):
#         if atoms is None:
#             atoms = self.atoms
#         self.calculate(atoms, properties=['hessian'])
#         hessian = self.results['hessian']
#         freqs, modes = np.linalg.eig(hessian)
#         modes = [modes[:,i].reshape(len(atoms),3)
#                  for i in range(len(freqs))]
#         return freqs, modes

import json
import time
import tensorflow as tf
import numpy as np
from ase.calculators.calculator import Calculator

import pinn.filters as filters
import pinn.layers as layers


class PINN(Calculator):
    def __init__(self, model=None):
        Calculator.__init__(self)
        if model is None:
            self.model = pinn_model()
        self.training_traj = None
        self.training_data = None
        self.calc_sess = None
        self.implemented_properties = ['energy', 'forces']

    def calculate(self, atoms=None, properties=['energy'], system_chages=None, sess=None):
        c_in, p_in, energy, c_flat = self.model.construct_running(atoms)
        # Properties to calculate
        results = {}
        if 'energy' in properties:
            results['energy'] = energy
        if 'forces' in properties:
            results['forces'] = tf.gradients(energy, c_in)
        if 'hessian' in properties:
            results['hessian'] = tf.hessians(energy, c_flat)[0]
        # Run the calculation
        c_mat = [atoms.get_positions()]
        p_mat = [self.model.p_filter.parse(atoms)]
        with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
            sess.run(tf.global_variables_initializer())
            self.results = sess.run(
                results, feed_dict={c_in: c_mat, p_in: p_mat})

    def get_vibrational_modes(self, atoms=None):
        if atoms is None:
            atoms = self.atoms
        self.calculate(atoms, properties=['hessian'])
        hessian = self.results['hessian']
        freqs, modes = np.linalg.eig(hessian)
        modes = [
            modes[:,i].reshape(len(atoms),3)
            for i in range(len(freqs))
        ]
        return freqs, modes

    def visualize(self, atoms):
        d_mat = np.expand_dims([atoms.get_all_distances()],3)
        p_mat = np.expand_dims(self.model.p_filter.parse(atoms),0)

        d_in, p_in, = self.model.construct_visualize(atoms)
        feed_dict = {d_in: d_mat, p_in: p_mat}
        with tf.Session() as sess:
            blocks = sess.run(blocks, feed_dict=feed_dict)
        return blocks

    def train(self, traj, optimizer=tf.train.AdamOptimizer(3e-4),
              batch_size=100, max_steps=100, log_interval=10, chkfile=None):
        tf.reset_default_graph()
        print('Processing input data', flush=True)
        dataset = self.model.parse_training_traj(traj)
        d_data = dataset['d_mat']
        p_data = dataset['p_mat']
        e_data = dataset['e_mat']*self.model.scale

        # Preparing the training model
        d_in, p_in, e_out = self.model.construct_training(
            batch_size, d_data.shape[1])
        e_in = tf.placeholder(self.model.dtype, shape=e_out.shape)
        cost = tf.nn.l2_loss(e_in - e_out)
        opt = optimizer.minimize(cost)
        n_batch = d_data.shape[0]//batch_size
        feed_dict = {}
        history = []

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for step in range(max_steps):
                perm = np.random.permutation(d_data.shape[0])
                for n in range(n_batch):
                    indices = perm[batch_size*n: batch_size*(n+1)]
                    feed_dict[d_in] = d_data[indices]
                    feed_dict[p_in] = p_data[indices]
                    feed_dict[e_in] = e_data[indices]
                    _, cost_now = sess.run([opt, cost], feed_dict=feed_dict)
                    history.append(np.sqrt(cost_now*2./batch_size))
                if (step + 1) % log_interval == 0:
                    for layer in self.model.layers:
                        layer.retrive_variables(sess, self.model.dtype)

                    if chkfile is not None:
                        self.model.save(chkfile)

                    epoch_now = np.array(history[-n_batch:])
                    if step>0:
                        epoch_prev = history[-n_batch*2:-n_batch]
                        dcost = np.mean(epoch_now) - np.mean(epoch_prev)
                    else:
                        dcost = np.nan

                    epoch_rms = np.sqrt(np.mean(np.square(
                        epoch_now-np.mean(epoch_now))))
                    print('Epoch %8i: Cost_avg=%10.4f, dCost=%10.4f RMS=%10.4f' %
                          (step+1, np.mean(epoch_now), dcost, epoch_rms), flush=True)

            # Run a last epoch to get the predictions
            e_predict = []
            for n in range(n_batch):
                indices=range(batch_size*n, batch_size*(n+1))
                e_predict.append(sess.run(e_out, feed_dict={d_in:d_data[indices],
                                                       p_in:p_data[indices]}))
        results = {
            'energy_data': e_data,
            'energy_predict': np.concatenate(e_predict),
            'history':history}
        return results

class pinn_model():
    '''
    '''

    def __init__(self,
                 dtype=tf.float32,
                 p_filter=filters.default_p_filter,
                 i_filter=filters.default_i_filter,
                 layers=layers.default_layers(),
                 dress_atoms=True,
                 atomic_dress=None):
        self.dtype = dtype
        self.dress_atoms = dress_atoms
        self.atomic_dress = atomic_dress
        self.p_filter = p_filter
        self.i_filter = i_filter
        self.layers = layers
        self.scale = 627.5

    def parse_training_traj(self, traj, en_data=None):
        dataset = {}
        n_max = max([len(atoms) for atoms in traj])

        d_mat_data = []
        p_mat_data = []
        for atoms in traj:
            pad = n_max - len(atoms)
            d_mat = atoms.get_all_distances()
            p_mat = self.p_filter.parse(atoms)
            d_mat_data.append(np.pad(d_mat, [[0, pad], [0, pad]], 'constant'))
            p_mat_data.append(np.pad(p_mat, [[0, pad], [0, 0]], 'constant'))

        e_mat_data = [atoms.get_potential_energy() for atoms in traj]

        dataset['d_mat'] = np.expand_dims(np.array(d_mat_data), 3)
        dataset['p_mat'] = np.array(p_mat_data)
        dataset['e_mat'] = np.array(e_mat_data)

        if self.dress_atoms:
            if self.atomic_dress is None:
                self.fit_atomic_dress(dataset['p_mat'], dataset['e_mat'])
            dataset['e_mat'] -= np.dot(np.sum(dataset['p_mat'], axis=1),
                                       self.atomic_dress)
        return dataset

    def fit_atomic_dress(self, p_mat, e_mat):
        print('Generating a new atomic dress')
        p_sum = np.sum(p_mat, axis=1)
        self.atomic_dress = np.linalg.lstsq(
            p_sum, e_mat)[0].tolist()


    def construct_training(self, batch_size, n_max):
        d_in = tf.placeholder(self.dtype, shape=(batch_size, n_max, n_max, 1))
        i_kernel, i_mask = self.i_filter.get_tensors(d_in)
        p_in, p_mask = self.p_filter.get_tensors(self.dtype, batch_size, n_max)
        e_out = self.construct_model(i_kernel, p_in, i_mask, p_mask)
        return d_in, p_in, e_out

    def construct_running(self, atoms):
        if atoms.pbc.any():
            raise('PBC is not currently supported')
        n_atoms = len(atoms)
        c_in = tf.placeholder(self.dtype, shape=(1, n_atoms, 3))
        # Reshape two times for the hessian
        c_flat = tf.reshape(c_in, [n_atoms*3])
        c_in = tf.reshape(c_flat, [1, n_atoms, 3])
        d_mat = tf.expand_dims(c_in, 1) - tf.expand_dims(c_in, 2)
        d_mat = tf.reduce_sum(tf.square(d_mat), axis=3, keep_dims=True)
        # To avoid NaN gradients due to sqrt
        d_mat = tf.where(d_mat > 0, d_mat, tf.zeros_like(d_mat)+1e-20)
        d_mat = tf.where(d_mat > 1e-19, tf.sqrt(d_mat), tf.zeros_like(d_mat))

        i_in, i_mask = self.i_filter.get_tensors(d_mat)
        p_in, p_mask = self.p_filter.get_tensors(self.dtype, 1, n_atoms)
        energy = self.construct_model(i_in, p_in, i_mask, p_mask)/self.scale
        if self.dress_atoms:
            energy = energy + tf.tensordot(tf.reduce_sum(p_in, axis=1),
                                           self.atomic_dress, [[1], [0]])
        energy = tf.squeeze(energy)
        return c_in, p_in, energy, c_flat

    def construct_model(self, i_kernel, p_in, i_mask, p_mask):
        max_order = max([layer.order for layer in self.layers])
        n, m = p_in.shape[0], p_in.shape[1]
        nodes = [p_in]
        for i in range(max_order):
            nodes.append(tf.constant(
                np.zeros([n]+[m]*(i+2)+[0]), dtype=self.dtype))

        masks = [p_mask, i_mask]
        for i in range(max_order-1):
            dim = i+3
            mask = tf.expand_dims(masks[-1],1)
            for d in range(dim-1):
                mask = mask & tf.expand_dims(masks[-1],d+2)
            masks.append(mask)

        tensors = {
            'kernel': i_kernel,
            'nodes': nodes,
            'masks': masks
        }
        e_out = 0
        for layer in self.layers:
            #print(layer)
            en = layer.process(tensors, self.dtype)
            if en is not None:
                e_out = e_out + en
            #print([[int(i) for i in node.shape] for node in tensors['nodes']])
        return e_out

    def load(self, fname):
        with open(fname, 'r') as f:
            model_dict = json.load(f)
        self.dtype = tf.__getattribute__(model_dict['dtype'])
        self.dress_atoms = model_dict['dress_atoms']
        self.atomic_dress = model_dict['atomic_dress']
        self.i_filter = filters.__getattribute__(
            model_dict['i_filter']['class'])(**model_dict['i_filter']['param'])
        self.p_filter = filters.__getattribute__(
            model_dict['p_filter']['class'])(**model_dict['p_filter']['param'])
        self.layers = [
            layers.__getattribute__(layer['class'])(**layer['param'])
            for layer in model_dict['layers']
        ]

    def save(self, fname):
        model_dict = self.__dict__.copy()
        model_dict['dtype'] = self.dtype.name
        model_dict['layers'] = [{'class': layer.__class__.__name__,
                                 'param': layer.__dict__}
                                for layer in self.layers]
        model_dict['i_filter'] = {'class': self.i_filter.__class__.__name__,
                                  'param': self.i_filter.__dict__}
        model_dict['p_filter'] = {'class': self.p_filter.__class__.__name__,
                                  'param': self.p_filter.__dict__}
        with open(fname, 'w') as f:
            json.dump(model_dict, f, indent=2)

import json
import time
import tensorflow as tf
import numpy as np
from ase.calculators.calculator import Calculator
import pinn.filters as filters
import pinn.layers as layers


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


    def train(self, traj, optimizer=tf.train.AdamOptimizer(3e-4),
              batch_size=100, max_steps=100, log_interval=10, chkfile=None):
        tf.reset_default_graph()
        print('Processing input data', flush=True)
        dataset = traj_parser(traj, self.p_filter)
        c_in = tf.placeholder(self.dtype,
                              shape=(batch_size, dataset.n_atoms, 3))
        # Preparing the training model
        tensors = self.get_tensors(c_in, training=True)
        e_out = tensors['energy']
        p_in = tensors['p_in']
        e_in = tf.placeholder(self.dtype, shape=e_out.shape)
        e_atom = tf.tensordot(tf.reduce_sum(p_in, axis=1),
                              self.atomic_dress, [[1], [0]])
        e_scaled = (e_in - e_atom) * self.scale

        cost = tf.nn.l2_loss(e_scaled - e_out)
        opt = optimizer.minimize(cost)
        n_batch = dataset.size//batch_size
        history = []

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for step in range(max_steps):
                perm = np.random.permutation(dataset.size)
                for n in range(n_batch):
                    indices = perm[batch_size*n: batch_size*(n+1)]
                    feed_dict = dataset.get_input(indices)
                    _, cost_now = sess.run([opt, cost],
                                           feed_dict={c_in: feed_dict['c_in'],
                                                      p_in: feed_dict['p_in'],
                                                      e_in: feed_dict['e_in']})
                    history.append(np.sqrt(cost_now*2./batch_size))
                if (step + 1) % log_interval == 0:
                    for layer in self.layers:
                        layer.retrive_variables(sess, self.dtype)

                    if chkfile is not None:
                        self.save(chkfile)

                    epoch_now = np.array(history[-n_batch:])
                    if step > 0:
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
                indices = range(batch_size*n, batch_size*(n+1))
                feed_dict = dataset.get_input(indices)
                e_predict.append(
                    sess.run(e_out, feed_dict={c_in: feed_dict['c_in'],
                                               p_in: feed_dict['p_in']}))
        results = {
            'energy_predict': np.concatenate(e_predict),
            'history': history}
        return results

    def fit_atomic_dress(self, p_mat, e_mat):
        print('Generating a new atomic dress')
        p_sum = np.sum(p_mat, axis=1)
        self.atomic_dress = np.linalg.lstsq(
            p_sum, e_mat)[0].tolist()

    def get_tensors(self, c_in, training=False):
        # Prepare the inputs
        n_image = c_in.shape[0]
        n_atoms = c_in.shape[1]
        d_mat = tf.expand_dims(c_in, 1) - tf.expand_dims(c_in, 2)
        d_mat = tf.reduce_sum(tf.square(d_mat), axis=3, keepdims=True)
        # To avould NaN gradients
        d_mat = tf.where(d_mat > 0, d_mat, tf.zeros_like(d_mat)+1e-20)
        d_mat = tf.where(d_mat > 1e-19, tf.sqrt(d_mat), tf.zeros_like(d_mat))

        # Construct the model
        i_kernel, i_mask = self.i_filter.get_tensors(d_mat)
        p_in, p_mask = self.p_filter.get_tensors(self.dtype, n_image, n_atoms)
        # Because we padded zeros when training
        i_mask = i_mask & (tf.expand_dims(p_mask, 1) & tf.expand_dims(p_mask, 2))
        max_order = max([layer.order for layer in self.layers])
        n, m = p_in.shape[0], p_in.shape[1]
        nodes = [p_in]
        for i in range(max_order):
            nodes.append(tf.constant(
                np.zeros([n]+[m]*(i+2)+[0]), dtype=self.dtype))

        masks = [p_mask, i_mask]
        for i in range(max_order-1):
            dim = i+3
            mask = tf.expand_dims(masks[-1], 1)
            for d in range(dim-1):
                mask = mask & tf.expand_dims(masks[-1], d+2)
            masks.append(mask)

        tensors = {
            'kernel': i_kernel,
            'nodes': nodes,
            'masks': masks
        }
        e_out = 0.0
        for layer in self.layers:
            en = layer.process(tensors, self.dtype)
            if en is not None:
                e_out = e_out + en
        # Atomic dress
        if not training:
            e_out = e_out/self.scale
            if self.dress_atoms:
                e_out = e_out + tf.tensordot(tf.reduce_sum(p_in, axis=1),
                                             self.atomic_dress, [[1], [0]])
        energy = tf.squeeze(e_out)
        # Outputs
        tensors = {'c_in': c_in, 'p_in': p_in, 'energy': energy}
        return tensors

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
            json.dump(model_dict, f)


class traj_parser():
    def __init__(self, traj, p_filter):
        self.size = len(traj)
        self.n_atoms = max([len(atoms) for atoms in traj])
        c_mat = []
        p_mat = []
        for atoms in traj:
            n_pad = self.n_atoms - len(atoms)
            c_mat.append(np.pad(atoms.get_positions(),
                                     [[0, n_pad], [0, 0]], 'constant'))
            p_mat.append(np.pad(p_filter.parse(atoms),
                                     [[0, n_pad], [0, 0]], 'constant'))
        self.c_mat = np.array(c_mat)
        self.p_mat = np.array(p_mat)
        self.e_mat = np.array([atoms.get_potential_energy() for atoms in traj])

    def get_input(self, index):
        c_in = self.c_mat[index]
        p_in = self.p_mat[index]
        e_in = self.e_mat[index]
        feed_dict = {'c_in': c_in,
                     'p_in': p_in,
                     'e_in': e_in}
        return feed_dict

class ani_parser():
    def __init__(self, fname, p_filter):
        self.size


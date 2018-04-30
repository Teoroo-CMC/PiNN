import json, os, time
import tensorflow as tf
import numpy as np
from ase.calculators.calculator import Calculator
from tensorflow.python.lib.io import file_io
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
                 atomic_dress=None):
        self.dtype = dtype

        self.p_filter = p_filter
        self.i_filter = i_filter
        if atomic_dress is None:
            atomic_dress = [0]*len(p_filter.element_list)
        self.atomic_dress = atomic_dress
        self.layers = layers
        self.scale = 627.5

    def get_energy(self, input, training=False):
        p_in = input['p_in']
        p_mask = input['p_mask']
        i_mask = input['i_mask']
        i_kernel = input['i_kernel']

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
        for i, layer in enumerate(self.layers):
            en = layer.process(tensors, self.dtype)

            if en is not None:
                e_out = e_out + en
        # Atomic dress
        if not training:
            e_out = e_out/self.scale
            if self.dress_atoms:
                e_out = e_out + tf.tensordot(tf.reduce_sum(p_in, axis=1),
                                             tf.constant(self.atomic_dress,dtype=self.dtype),
                                             [[1], [0]])
        energy = tf.squeeze(e_out)
        # Outputs
        return energy

    def get_inputs(self, data):
        c_in = data['c_in']
        p_in = data['p_in']
        d_mat = coord_to_dist(c_in)
        i_kernel, i_mask = self.i_filter.get_tensors(d_mat)
        p_mask = tf.reduce_sum(p_in, axis=-1, keepdims=True) > 0
        # Because we padded zeros
        i_mask = i_mask & (tf.expand_dims(p_mask, 1) &
                           tf.expand_dims(p_mask, 2))
        data['p_mask'] = p_mask
        data['i_mask'] = i_mask
        data['i_kernel'] = i_kernel
        return data

    def train(self, dataset,
              max_epoch=10, max_steps=100,
              batch_size=100, learning_rate=3e-4,
              log_dir='logs', log_interval=10,
              chk_dir='chks', chk_interval=100,
              job_name='training'):
        tf.reset_default_graph()
        print('Building the model', flush=True)
        # Preparing the training model
        optimizer=tf.train.AdamOptimizer(learning_rate)
        dtypes = {'c_in': self.dtype, 'a_in': tf.int32, 'e_in': self.dtype}
        dshapes = {'c_in': [dataset.n_atoms, 3],
                   'a_in': [dataset.n_atoms],
                   'e_in': [1]}
        dataset = dataset.get_training(dtypes)
        #dataset = dataset.map(lambda data: self.p_filter.parse(data, self.dtype))
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(
            100000, max_epoch))
        dataset = dataset.apply(
            tf.contrib.data.padded_batch_and_drop_remainder(
                batch_size, dshapes))

        #dataset = dataset.map(self.get_inputs)
        dataset = dataset.prefetch(1000)
        iterator = dataset.make_one_shot_iterator()
        input = iterator.get_next()
        input = self.p_filter.parse(input, self.dtype)
        input = self.get_inputs(input)



        e_out = self.get_energy(input, training=True)
        e_atom = tf.tensordot(tf.reduce_sum(input['p_in'], axis=1),
                              tf.constant(self.atomic_dress, self.dtype),
                              [[1], [0]])

        e_in = (tf.squeeze(input['e_in']) - e_atom) * self.scale
        cost = tf.losses.mean_squared_error(e_in, e_out)
        opt = optimizer.minimize(cost)
        tf.summary.scalar('batch_cost', tf.sqrt(cost))
        tf.summary.histogram('batch_hist', e_in - e_out)
        tf.summary.histogram('batch_e_in', e_in)
        tf.summary.histogram('batch_e_out', e_out)
        merged = tf.summary.merge_all()
        run_name = time.strftime("{}-%m%d%H%M".format(job_name))
        chk_name = os.path.join(chk_dir,'{}-chk.json'.format(job_name))

        print('Start training', flush=True)
        with tf.Session() as sess:
            log_writer = tf.summary.FileWriter(
                os.path.join(log_dir, run_name))

            sess.run(tf.global_variables_initializer())

            for step in range(int(max_steps)):
                try:
                    _, summary = sess.run([opt, merged])
                    if (step + 1) % log_interval == 0:
                        log_writer.add_summary(summary, step+1)
                    if (step + 1) % chk_interval == 0:
                        for layer in self.layers:
                            layer.retrive_variables(sess, self.dtype)
                        print('Saving {} (step={})'.format(chk_name, step+1), flush=True)
                        self.save(chk_name)
                except tf.errors.OutOfRangeError:
                    print('End of epoches', flush=True)
                    break

    def fit_atomic_dress(self):
        print('Generating a new atomic dress')
        p_sum = np.sum(p_mat, axis=1)
        self.atomic_dress = np.linalg.lstsq(
            p_sum, e_mat)[0].tolist()

    def load(self, fname):
        with file_io.FileIO(fname, 'r') as f:
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
        with file_io.FileIO(fname, 'w') as f:
            json.dump(model_dict, f)


def coord_to_dist(c_mat, pbc=False):
    d_mat = tf.expand_dims(c_mat, 1) - tf.expand_dims(c_mat, 2)
    d_mat = tf.reduce_sum(tf.square(d_mat), axis=3, keepdims=True)
    # To avould NaN gradients
    d_mat = tf.where(d_mat > 0, d_mat, tf.zeros_like(d_mat)+1e-20)
    d_mat = tf.where(d_mat > 1e-19, tf.sqrt(d_mat), tf.zeros_like(d_mat))
    return d_mat

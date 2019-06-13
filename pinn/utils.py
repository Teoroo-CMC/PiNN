import tensorflow as tf
import numpy as np
from functools import wraps

def get_atomic_dress(dataset, elems, max_iter=1000):
    """
    Fit the atomic energy with a element dependent atomic dress
    
    Args:
         dataset: dataset to fit
         elems: a list of element numbers
    Returns:
         atomic_dress: a dictionary comprising the atomic energy of each element
         error: residue error of the atomic dress
    """
    tensors = dataset.make_one_shot_iterator().get_next()
    if 'ind_1' not in tensors:
        tensors['ind_1'] = tf.expand_dims(tf.zeros_like(tensors['elems']),1)
        tensors['e_data'] = tf.expand_dims(tensors['e_data'],0)
    count = tf.equal(tf.expand_dims(tensors['elems'],1), tf.expand_dims(elems, 0))
    count = tf.cast(count, tf.int32)
    count = tf.segment_sum(count, tensors['ind_1'][:,0])
    sess = tf.Session()
    x, y = [],[]
    for i in range(max_iter):
        try:
            x_i, y_i = sess.run((count,tensors['e_data']))
            x.append(x_i)
            y.append(y_i)
        except tf.errors.OutOfRangeError:
            break
    x, y = np.concatenate(x, 0), np.concatenate(y, 0)
    beta = np.dot(np.dot(np.linalg.pinv(np.dot(x.T, x)),x.T),np.array(y))
    dress = {e:float(beta[i]) for (i, e) in enumerate(elems)}
    error = np.dot(x, beta) - y
    return dress, error

def pi_named(default_name='unnamed'):
    """Decorate a layer to have a name """
    def decorator(func):
        @wraps(func)
        def named_layer(*args, name=default_name, **kwargs):
            with tf.variable_scope(name):
                return func(*args, **kwargs)
        return named_layer
    return decorator

def pinn_filter(func):
    @wraps(func)
    def filter_wrapper(*args, **kwargs):
        return lambda t: func(t, *args, **kwargs)
    return filter_wrapper

def TuneTrainable(trainer_fn):
    """Helper function for geting a trainable to use with Tune
    
    The function expectes a trainer_fn function which takes a config as input,
    and returns four items.

    - model: the tensorflow estimator.
    - train_spec: training specification.
    - eval_spec: evaluation specification.
    - reporter: a function which returns the metrics given evalution.

    The resulting trainable reports the metrics when checkpoints are saved,
    the report frequency is controlled by the checkpoint frequency,
    and the metrics are determined by reporter.
    """
    import os 
    from ray.tune import Trainable
    from tensorflow.train import CheckpointSaverListener
    class _tuneStoper(CheckpointSaverListener):
        def after_save(self, session, global_step_value):
            return True
    class TuneTrainable(Trainable):
        def _setup(self, config):
            tf.logging.set_verbosity(tf.logging.ERROR)
            self.config = config
            model, train_spec, eval_spec, reporter = trainer_fn(config)
            self.model = model
            self.train_spec = train_spec
            self.eval_spec = eval_spec
            self.reporter = reporter
        
        def _train(self):
            import warnings
            index_warning = 'Converting sparse IndexedSlices'
            warnings.filterwarnings('ignore', index_warning)
            model = self.model
            model.train(input_fn=self.train_spec.input_fn,
                        max_steps=self.train_spec.max_steps,
                        hooks=self.train_spec.hooks,
                        saving_listeners=[_tuneStoper()])
            eval_out = model.evaluate(input_fn=self.eval_spec.input_fn,
                                     steps=self.eval_spec.steps,
                                     hooks=self.eval_spec.hooks)
            metrics = self.reporter(eval_out)
            return metrics

        def _save(self, checkpoint_dir):
            latest_checkpoint = self.model.latest_checkpoint()
            chkpath = os.path.join(checkpoint_dir, 'path.txt')
            with open(chkpath, 'w') as f:
                f.write(latest_checkpoint)
            return chkpath
    
        def _restore(self, checkpoint_path):
            with open(checkpoint_path) as f:
                chkpath = f.readline().strip()
            self.model, _, _, _ = trainer_fn(self.config, chkpath)
    return TuneTrainable

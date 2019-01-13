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
    sess = tf.Session()
    x, y = [],[]
    for i in range(max_iter):
        try:
            batch = sess.run(tensors)
            x.append(batch['atoms'])
            y.append(batch['e_data'])
            
        except tf.errors.OutOfRangeError:
            break
    if len(x[0].shape)==1:
        x, y = np.array(x), np.array(y)
    else:
        x, y = np.concatenate(x, 0), np.concatenate(y, 0)
    x = np.sum(np.expand_dims(x,2)==np.reshape(elems, [1,1,len(elems)]),1)
    beta = np.dot(np.dot(np.linalg.pinv(np.dot(x.T, x)),x.T),np.array(y))
    dress = {e:beta[i] for (i, e) in enumerate(elems)}
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

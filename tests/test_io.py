import pytest
import tensorflow as tf
import numpy as np
from tensorflow.errors import OutOfRangeError
from helpers import *

# Some simple test that dataset can be loaded

def test_numpy():
    dataset = get_trivial_numpy_ds()
    data = get_trivial_numpy()
    item = dataset.make_one_shot_iterator().get_next()
    with tf.Session() as sess:
        out = sess.run(item)
        for k in out.keys():
            assert_almost_equal(out[k], data[k])
        with pytest.raises(OutOfRangeError):
            out = sess.run(item)

def test_qm9():
    dataset = get_trivial_qm9_ds()
    data = get_trivial_numpy()
    item = dataset.make_one_shot_iterator().get_next()
    with tf.Session() as sess:
        out = sess.run(item)
        for k in out.keys():
            assert_almost_equal(out[k], data[k])
        with pytest.raises(OutOfRangeError):
            out = sess.run(item)

def test_runner():
    dataset = get_trivial_runner_ds()
    bohr2ang = 0.5291772109
    data = get_trivial_numpy()
    data['coord'] *= bohr2ang
    data['cell'] *= bohr2ang
    item = dataset.make_one_shot_iterator().get_next()
    with tf.Session() as sess:
        out = sess.run(item)
        for k in out.keys():
            if k in data: # runner has many labels
                assert_almost_equal(out[k], data[k])
        with pytest.raises(OutOfRangeError):
            out = sess.run(item)
    
def test_split():
    # Test that dataset is splitted according to the given ratio
    from pinn.io import load_numpy
    data = get_trivial_numpy()
    data = {k: np.stack([[v]]*10, axis=0) for k,v in data.items()}
    dataset = load_numpy(data, split={'train': 8, 'test':2})
    train = dataset['train'].make_one_shot_iterator().get_next()
    test = dataset['test'].make_one_shot_iterator().get_next()    
    with tf.Session() as sess:
        for i in range(8):
            out = sess.run(train)
        with pytest.raises(OutOfRangeError):
            out = sess.run(train)
        for i in range(2):
            out = sess.run(test)
        with pytest.raises(OutOfRangeError):
            out = sess.run(test)            

    
def test_write():
    from pinn.io import load_tfrecord, write_tfrecord, sparse_batch
    ds = get_trivial_runner_ds().repeat(20)
    write_tfrecord('test.yml', ds)
    ds_tfr = load_tfrecord('test.yml')

    ds_batch = ds.apply(sparse_batch(20))
    write_tfrecord('test_batch.yml', ds_batch)
    ds_batch_tfr = load_tfrecord('test_batch.yml')

    with tf.Session() as sess:
        label = sess.run(ds.make_one_shot_iterator().get_next())
        out = sess.run(ds_tfr.make_one_shot_iterator().get_next())
        for k in out.keys():
            assert_almost_equal(label[k], out[k])

        label = sess.run(ds_batch.make_one_shot_iterator().get_next())
        out = sess.run(ds_batch_tfr.make_one_shot_iterator().get_next())
        for k in out.keys():
            assert_almost_equal(label[k], out[k])        


    
    

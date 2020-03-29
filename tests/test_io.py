# -*- coding: utf-8 -*-
"""Some simple test that dataset can be loaded"""

import pytest, tempfile
import tensorflow as tf
import numpy as np
from helpers import *


def test_numpy():
    dataset = get_trivial_numpy_ds()
    data = get_trivial_numpy()
    iterator = iter(dataset)
    out = next(iterator)
    for k in out.keys():
        assert np.allclose(out[k], data[k])
        with pytest.raises(StopIteration):
            out = next(iterator)


def test_qm9():
    dataset = get_trivial_qm9_ds()
    data = get_trivial_numpy()
    iterator = iter(dataset)
    out = next(iterator)
    for k in out.keys():
        assert np.allclose(out[k], data[k])
        with pytest.raises(StopIteration):
            out = next(iterator)


def test_runner():
    dataset = get_trivial_runner_ds()
    bohr2ang = 0.5291772109
    data = get_trivial_numpy()
    data['coord'] *= bohr2ang
    data['cell'] *= bohr2ang
    iterator = iter(dataset)
    out = next(iterator)
    for k in data.keys(): # RuNNer has many labels, we do not test all of them
        assert np.allclose(out[k], data[k])
        with pytest.raises(StopIteration):
            out = next(iterator)


def test_split():
    # Test that dataset is splitted according to the given ratio
    from pinn.io import load_numpy
    data = get_trivial_numpy()
    data = {k: np.stack([[v]]*10, axis=0) for k, v in data.items()}
    dataset = load_numpy(data, split={'train': 8, 'test': 2})
    train = iter(dataset['train'])
    test = iter(dataset['test'])
    for i in range(8):
        out = next(train)
    with pytest.raises(StopIteration):
        out = next(train)
    for i in range(2):
        out = next(test)
    with pytest.raises(StopIteration):
        out = next(test)


def test_write():
    import os
    from pinn.io import load_tfrecord, write_tfrecord, sparse_batch
    from shutil import rmtree
    tmp = tempfile.mkdtemp(prefix='pinn_test')
    ds = get_trivial_runner_ds().repeat(20)
    write_tfrecord('{}/test.yml'.format(tmp), ds)
    ds_tfr = load_tfrecord('{}/test.yml'.format(tmp))

    ds_batch = ds.apply(sparse_batch(20))
    write_tfrecord('{}/test_batch.yml'.format(tmp), ds_batch)
    ds_batch_tfr = load_tfrecord('{}/test_batch.yml'.format(tmp))

    label = next(iter(ds))
    out = next(iter(ds_tfr))
    for k in out.keys():
        assert np.allclose(label[k], out[k])

    label = next(iter(ds_batch))
    out = next(iter(ds_batch_tfr))
    for k in out.keys():
        assert np.allclose(label[k], out[k])
    rmtree(tmp, ignore_errors=True)

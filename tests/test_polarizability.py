# -*- coding: utf-8 -*-
"""Numerical tests for virials and energy conservation of potentials"""
import tempfile
import pytest
import numpy as np
import tensorflow as tf
from pinn.io import sparse_batch
from shutil import rmtree
import yaml
from pinn.io import load_qm7b
from pinn.models import pol_models
import pinn
import tensorflow as tf
from pinn import get_network
import os

@pytest.mark.forked
def test_pol_eem():
    testpath = tempfile.mkdtemp()
    network_params = {
        'ii_nodes': [8, 8],
        'pi_nodes': [8, 8],
        'pp_nodes': [8, 8],
        'out_nodes': [8, 8],
        'depth': 3,
        'rc': 5.,
        'n_basis': 5,
        'atom_types': [1,6]
    }
    params = {
        'model_dir': testpath,
        'network': {
            'name': 'PiNet2',
            'params': network_params},
        'model': {
            'name': 'pol_eem_model',
            'params': {}}}
    _pol_test(params)
    rmtree(testpath)
    
@pytest.mark.forked
def test_pol_acks2():
    testpath = tempfile.mkdtemp()
    network_params = {
        'ii_nodes': [8, 8],
        'pi_nodes': [8, 8],
        'pp_nodes': [8, 8],
        'out_nodes': [8, 8],
        'depth': 3,
        'rc': 5.,
        'n_basis': 5,
        'atom_types': [1,6]
    }
    params = {
        'model_dir': testpath,
        'network': {
            'name': 'PiNet2',
            'params': network_params},
        'model': {
            'name': 'pol_acks2_model',
            'params': {}}}
    _pol_test(params)
    rmtree(testpath)
    
@pytest.mark.forked
def test_pol_etainv():
    testpath = tempfile.mkdtemp()
    network_params = {
        'ii_nodes': [8, 8],
        'pi_nodes': [8, 8],
        'pp_nodes': [8, 8],
        'out_nodes': [8, 8],
        'depth': 3,
        'rc': 5.,
        'n_basis': 5,
        'atom_types': [1,6]
    }
    params = {
        'model_dir': testpath,
        'network': {
            'name': 'PiNet2',
            'params': network_params},
        'model': {
            'name': 'pol_etainv_model',
            'params': {}}}
    _pol_test(params)
    rmtree(testpath)

@pytest.mark.forked
def test_pol_local():
    testpath = tempfile.mkdtemp()
    network_params = {
        'ii_nodes': [8, 8],
        'pi_nodes': [8, 8],
        'pp_nodes': [8, 8],
        'out_nodes': [8, 8],
        'depth': 3,
        'rc': 5.,
        'n_basis': 5,
        'atom_types': [1,6]
    }
    params = {
        'model_dir': testpath,
        'network': {
            'name': 'PiNet2',
            'params': network_params},
        'model': {
            'name': 'pol_local_model',
            'params': {}}}
    _pol_test(params)
    rmtree(testpath)
    
def pre_fn(tensors):
    network_params = {
        'ii_nodes': [8, 8],
        'pi_nodes': [8, 8],
        'pp_nodes': [8, 8],
        'out_nodes': [8, 8],
        'depth': 3,
        'rc': 5.,
        'n_basis': 5,
        'atom_types': [1,6]
    }
    params = {
        'network': {
            'name': 'PiNet2',
            'params': network_params}}
    with tf.name_scope("PRE") as scope:
        network = get_network(params['network'])
        tensors = network.preprocess(tensors)
    return tensors
    
def _pol_test(params):
    import numpy as np
    from glob import glob
    this_dir = os.path.dirname(os.path.abspath(__file__))
    file = '{}/examples/pol_data/test.xyz'.format(this_dir)
    model = pinn.get_model(params)
    dataset = lambda: load_qm7b(file)
    train = lambda: dataset().apply(sparse_batch(10)).map(pre_fn).cache().repeat().shuffle(1000)
    test = lambda: dataset().apply(sparse_batch(10)).map(pre_fn)
    data= list(dataset())
    train_spec = tf.estimator.TrainSpec(input_fn=train, max_steps=1000)
    eval_spec = tf.estimator.EvalSpec(input_fn=test, throttle_secs=600)
    results, _ = tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    p_pred=[out['alpha'] for out in model.predict(lambda: dataset().apply(sparse_batch(1)))]
    nats=len(data[0]['elems'])
    assert np.allclose(results['METRICS/alpha_per_atom_RMSE'],
                        np.sqrt(np.mean((p_pred[0]- data[0]['ptensor'])**2)*9)/nats, rtol=1e-2)
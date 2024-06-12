# -*- coding: utf-8 -*-
import numpy as np
import tempfile
import pytest
import tensorflow as tf
from pinn.io import load_numpy, sparse_batch
from shutil import rmtree
def _get_dipole_data():
    from ase.build import molecule

    water = molecule('H2O')

    q_O = -0.8476
    q_H = abs(q_O)/2
    q = np.array([[q_O, q_H, q_H]])

    coord, elems, d_data = [], [], []
    for i in range(120):
        water.rotate(3, 'x')
        r = water.positions.copy()
        dipole = q @ r

        coord.append(r)
        elems.append(water.numbers)
        d_data.append(dipole[0])

    data = {
            'coord': np.array(coord),
            'elems': np.array(elems),
            'd_data': np.array(d_data)
            }

    return data


@pytest.mark.forked
def test_pinn_atomic_dipole():
    testpath = tempfile.mkdtemp()
    network_params = {
        'ii_nodes': [8, 8],
        'pi_nodes': [8, 8],
        'pp_nodes': [8, 8],
        'out_nodes': [8, 8],
        'depth': 3,
        'rc': 5.,
        'n_basis': 5,
        'atom_types': [1, 8]
    }
    params = {
        'model_dir': testpath,
        'network': {
            'name': 'PiNet2',
            'params': network_params},
        'model': {
            'name': 'combined_dipole_model',
            'params':  {
                'd_scale': 1.0,
                'd_unit': 1.0
                }
            }}
    _atomic_dipole_tests(params)
    rmtree(testpath)



def _atomic_dipole_tests(params):
    # Series of tasks that the atomic dipole model should pass
    import pinn

    data = _get_dipole_data()

    def train(): return load_numpy(data).repeat().shuffle(
        500).apply(sparse_batch(50))

    def test(): return load_numpy(data).apply(sparse_batch(10))
    train_spec = tf.estimator.TrainSpec(input_fn=train, max_steps=1e3)
    eval_spec = tf.estimator.EvalSpec(input_fn=test, steps=100)

    model = pinn.get_model(params)
    results, _ = tf.estimator.train_and_evaluate(model, train_spec, eval_spec)

    #assert False, results
    


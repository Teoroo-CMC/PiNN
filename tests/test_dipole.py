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
        r = water.positions
        dipole = q @ r

        coord.append(water.positions)
        elems.append(water.numbers)
        d_data.append(dipole.flatten())

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
        'atom_types': [1]
    }
    params = {
        'model_dir': testpath,
        'network': {
            'name': 'PiNet',
            'params': network_params},
        'model': {
            'name': 'atomic_dipole_model'}}
    _atomic_dipole_tests(params)
    rmtree(testpath)



def _atomic_dipole_tests(params):
    # Series of tasks that the atomic dipole model should pass
    import pinn
    from pinn.calculator import PiNN_calc

    data = _get_dipole_data()

    def train(): return load_numpy(data).repeat().shuffle(
        500).apply(sparse_batch(50))

    def test(): return load_numpy(data).apply(sparse_batch(10))
    train_spec = tf.estimator.TrainSpec(input_fn=train, max_steps=1e3)
    eval_spec = tf.estimator.EvalSpec(input_fn=test, steps=100)

    model = pinn.get_model(params)
    results, _ = tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    
    assert np.allclose(results['METRICS/E_RMSE']/params['model']['params']['e_scale'],
                       np.sqrt(np.mean((e_pred/params['model']['params']['e_unit']
                                        - data['e_data'])**2)), rtol=1e-2)


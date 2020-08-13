# -*- coding: utf-8 -*-
"""Numerical tests for virials and energy conservation of potentials"""
import tempfile
import pytest
import numpy as np
import tensorflow as tf
from pinn.io import load_numpy, sparse_batch
from shutil import rmtree
from ase import Atoms

@pytest.mark.forked
def test_pinn_potential():
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
            'name': 'potential_model',
            'params': {
                'use_force': True,
                'e_dress': {1: 0.5},
                'e_scale': 5.0,
                'e_unit': 2.0}}}
    _potential_tests(params)
    rmtree(testpath)


@pytest.mark.forked
def test_bpnn_potential():
    testpath = tempfile.mkdtemp()
    network_params = {
        'sf_spec': [
            {'type': 'G2', 'i': 1, 'j': 1, 'eta': [
                0.1, 0.1, 0.1], 'Rs': [1., 2., 3.]},
            {'type': 'G3', 'i': 1, 'j': 1, 'k': 1,
             'eta': [0.1, 0.1, 0.1, 0.1], 'lambd': [1., 1., -1., -1.], 'zeta':[1., 1., 4., 4.]},
            {'type': 'G4', 'i': 1, 'j': 1, 'k': 1,
             'eta': [0.1, 0.1, 0.1, 0.1], 'lambd': [1., 1., -1., -1.], 'zeta':[1., 1., 4., 4.]}
        ],
        'nn_spec': {1: [8, 8]},
        'rc': 5.,
    }
    params = {
        'model_dir': testpath,
        'network': {
            'name': 'BPNN',
            'params': network_params},
        'model': {
            'name': 'potential_model',
            'params': {
                'use_force': True,
                'e_dress': {1: 0.5},
                'e_scale': 5.0,
                'e_unit': 2.0}}}
    _potential_tests(params)
    rmtree(testpath)


@pytest.mark.forked
def test_bpnn_potential_pre_cond():
    from pinn.networks import BPNN

    testpath = tempfile.mkdtemp()
    network_params = {
        'sf_spec': [
            {'type': 'G2', 'i': 1, 'j': 1, 'eta': [
                0.1, 0.1, 0.1], 'Rs': [1., 2., 3.]},
            {'type': 'G3', 'i': 1, 'j': 1, 'k': 1,
             'eta': [0.1, 0.1, 0.1, 0.1], 'lambd': [1., 1., -1., -1.], 'zeta':[1., 1., 4., 4.]},
            {'type': 'G4', 'i': 1, 'j': 1, 'k': 1,
             'eta': [0.1, 0.1, 0.1, 0.1], 'lambd': [1., 1., -1., -1.], 'zeta':[1., 1., 4., 4.]}
        ],
        'nn_spec': {1: [8, 8]},
        'rc': 5.
    }
    bpnn =  BPNN(**network_params)
    dataset = load_numpy(_get_lj_data(), split=1)\
        .apply(sparse_batch(10)).map(bpnn.preprocess)

    batches = [tensors for tensors in dataset]
    fp_range = []
    for i in range(len(network_params['sf_spec'])):
        fp_max = max([b[f'fp_{i}'].numpy().max() for b in batches])
        fp_min = max([b[f'fp_{i}'].numpy().min() for b in batches])
        fp_range.append([float(fp_min), float(fp_max)])

    network_params['fp_scale'] = True
    network_params['fp_range'] = fp_range
    params = {
        'model_dir': testpath,
        'network': {
            'name': 'BPNN',
            'params': network_params},
        'model': {
            'name': 'potential_model',
            'params': {
                'use_force': True,
                'e_dress': {1: 0.5},
                'e_scale': 5.0,
                'e_unit': 2.0}}}
    _potential_tests(params)
    rmtree(testpath)


def _get_lj_data():
    from ase.calculators.lj import LennardJones

    atoms = Atoms('H3', positions=[[0, 0, 0], [0, 1, 0], [1, 1, 0]])
    atoms.set_calculator(LennardJones(rc=5.0))
    coord, elems, e_data, f_data = [], [], [], []
    for x_a in np.linspace(-5, 0, 1000):
        atoms.positions[0, 0] = x_a
        coord.append(atoms.positions.copy())
        elems.append(atoms.numbers)
        e_data.append(atoms.get_potential_energy())
        f_data.append(atoms.get_forces())

    data = {
        'coord': np.array(coord),
        'elems': np.array(elems),
        'e_data': np.array(e_data),
        'f_data': np.array(f_data)
    }
    return data


def _potential_tests(params):
    # Series of tasks that a potential should pass
    import pinn
    from pinn.calculator import PiNN_calc

    data = _get_lj_data()

    def train(): return load_numpy(data, split=1).repeat().shuffle(
        500).apply(sparse_batch(50))

    def test(): return load_numpy(data, split=1).apply(sparse_batch(10))
    train_spec = tf.estimator.TrainSpec(input_fn=train, max_steps=1e3)
    eval_spec = tf.estimator.EvalSpec(input_fn=test, steps=100)

    model = pinn.get_model(params)
    results, _ = tf.estimator.train_and_evaluate(model, train_spec, eval_spec)

    # The calculator should be accessable with model_dir
    atoms = Atoms('H3', positions=[[0, 0, 0], [0, 1, 0], [1, 1, 0]])
    calc = pinn.get_calc(params, properties=['energy', 'forces', 'stress'])

    # Test energy dress and scaling
    # Make sure we have the correct error reports
    e_pred, f_pred = [], []
    for coord in data['coord']:
        atoms.positions = coord
        calc.calculate(atoms)
        e_pred.append(calc.get_potential_energy())
        f_pred.append(calc.get_forces())

    f_pred = np.array(f_pred)
    e_pred = np.array(e_pred)

    assert np.allclose(results['METRICS/F_RMSE']/params['model']['params']['e_scale'],
                        np.sqrt(np.mean((f_pred/params['model']['params']['e_unit']
                                         - data['f_data'])**2)), rtol=5e-3)
    assert np.allclose(results['METRICS/E_RMSE']/params['model']['params']['e_scale'],
                       np.sqrt(np.mean((e_pred/params['model']['params']['e_unit']
                                        - data['e_data'])**2)), rtol=5e-3)

    # Test energy conservation
    e_pred, f_pred = [], []
    x_a_range = np.linspace(-6, -3, 500)
    for x_a in np.linspace(-6, -3, 500):
        atoms.positions[0, 0] = x_a
        calc.calculate(atoms)
        e_pred.append(calc.get_potential_energy())
        f_pred.append(calc.get_forces())
    e_pred = np.array(e_pred)
    f_pred = np.array(f_pred)

    de = e_pred[-1] - e_pred[0]
    int_f = np.trapz(f_pred[:, 0, 0], x=x_a_range)
    print(f_pred)
    assert np.allclose(de, -int_f, rtol=5e-3)

    # Test virial pressure
    e_pred, p_pred = [], []
    l_range = np.linspace(3, 3.5, 500)
    atoms.positions[0, 0] = 0
    atoms.set_cell([3, 3, 3])
    atoms.set_pbc(True)
    for l in l_range:
        atoms.set_cell([l, l, l], scale_atoms=True)
        calc.calculate(atoms)
        e_pred.append(calc.get_potential_energy())
        p_pred.append(np.sum(calc.get_stress()[:3])/3)

    de = e_pred[-1] - e_pred[0]
    int_p = np.trapz(p_pred, x=l_range**3)
    assert np.allclose(de, int_p, rtol=5e-3)

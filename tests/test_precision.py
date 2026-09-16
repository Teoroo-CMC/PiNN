# -*- coding: utf-8 -*-
"""settings.dtype (YAML) and calculator default_dtype (constructor)."""
import os
import tempfile

import pytest
import tensorflow as tf
import yaml
from tensorflow.python.lib.io.file_io import FileIO

from pinn.calculator import PiNN_calc
from pinn.models.base import (
    apply_dtype, dtype_from_params, tf_dtype_from_name,
)


def test_tf_dtype_from_name():
    assert tf_dtype_from_name('float32') == tf.float32
    assert tf_dtype_from_name('float64') == tf.float64
    assert tf_dtype_from_name(None) == tf.float32
    with pytest.raises(ValueError, match='Unknown dtype'):
        tf_dtype_from_name('float16')
    with pytest.raises(ValueError, match='Unknown dtype'):
        tf_dtype_from_name('fp64')


def test_dtype_from_params():
    assert dtype_from_params({}) == 'float32'
    assert dtype_from_params({'settings': None}) == 'float32'
    assert dtype_from_params({'settings': {'dtype': 'float64'}}) == 'float64'


def test_calc_default_dtype_is_constructor_only():
    class _Model:
        params = {'settings': {'dtype': 'float64'}}

    calc = PiNN_calc(model=_Model())
    assert calc._dtype_name() == 'float64'
    assert calc._tf_dtype() == tf.float64
    calc_override = PiNN_calc(model=_Model(), default_dtype='float32')
    assert calc_override._tf_dtype() == tf.float32
    calc_plain = PiNN_calc(model=type('M', (), {})())
    assert calc_plain._tf_dtype() == tf.float32


@pytest.mark.forked
def test_unknown_dtype_raises():
    with pytest.raises(ValueError, match='Unknown dtype'):
        apply_dtype('fp16')


@pytest.mark.forked
def test_float32_is_the_default():
    apply_dtype('float32')
    assert tf.keras.backend.floatx() == 'float32'
    apply_dtype(None)
    assert tf.keras.backend.floatx() == 'float32'


@pytest.mark.parametrize('name', ['float32', 'float64'])
@pytest.mark.forked
def test_apply_dtype(name):
    apply_dtype(name)
    assert tf.keras.backend.floatx() == name
    apply_dtype('float32')


@pytest.mark.forked
def test_default_params_record_dtype():
    import pinn
    testpath = tempfile.mkdtemp()
    params = {
        'model_dir': testpath,
        'network': {
            'name': 'PiNet',
            'params': {
                'ii_nodes': [4, 4],
                'pi_nodes': [4, 4],
                'pp_nodes': [4, 4],
                'out_nodes': [4],
                'depth': 2,
                'rc': 4.0,
                'n_basis': 4,
                'atom_types': [1]}},
        'model': {
            'name': 'potential_model',
            'params': {'use_force': False}}}
    pinn.get_model(params)
    with FileIO(os.path.join(testpath, 'params.yml'), 'r') as f:
        saved = yaml.load(f, Loader=yaml.Loader)
    assert saved.get('settings', {}).get('dtype', 'float32') == 'float32'
    assert 'default_dtype' not in saved.get('settings', {})
    apply_dtype('float32')


@pytest.mark.forked
def test_float64_trains():
    import numpy as np
    import pinn
    from pinn.io import load_numpy, sparse_batch

    testpath = tempfile.mkdtemp()
    n = 8
    data = {
        'coord': np.random.randn(n, 3, 3).astype(np.float64),
        'elems': np.ones((n, 3), dtype=np.int32),
        'e_data': np.random.randn(n).astype(np.float64),
    }
    params = {
        'model_dir': testpath,
        'settings': {'dtype': 'float64'},
        'network': {
            'name': 'PiNet',
            'params': {
                'ii_nodes': [4, 4],
                'pi_nodes': [4, 4],
                'pp_nodes': [4, 4],
                'out_nodes': [4],
                'depth': 2,
                'rc': 4.0,
                'n_basis': 4,
                'atom_types': [1]}},
        'model': {
            'name': 'potential_model',
            'params': {'use_force': False}}}

    def train():
        return load_numpy(data).repeat().shuffle(n).apply(sparse_batch(4))

    def test():
        return load_numpy(data).apply(sparse_batch(4))

    model = pinn.get_model(params)
    train_spec = tf.estimator.TrainSpec(input_fn=train, max_steps=2)
    eval_spec = tf.estimator.EvalSpec(input_fn=test, steps=1)
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    with FileIO(os.path.join(testpath, 'params.yml'), 'r') as f:
        saved = yaml.load(f, Loader=yaml.Loader)
    assert saved['settings']['dtype'] == 'float64'
    apply_dtype('float32')


@pytest.mark.forked
def test_calc_predicts_across_dtypes():
    """A float32 checkpoint served at float64 (and vice versa).

    The Estimator finalizes the graph before the scaffold saver restores, so
    the cross-dtype restore must not create new ops.
    """
    import numpy as np
    import pinn
    from ase import Atoms
    from pinn.io import load_numpy, sparse_batch

    testpath = tempfile.mkdtemp()
    n = 8
    data = {
        'coord': np.random.randn(n, 3, 3).astype(np.float32),
        'elems': np.ones((n, 3), dtype=np.int32),
        'e_data': np.random.randn(n).astype(np.float32),
    }
    params = {
        'model_dir': testpath,
        'network': {
            'name': 'PiNet',
            'params': {
                'ii_nodes': [4, 4],
                'pi_nodes': [4, 4],
                'pp_nodes': [4, 4],
                'out_nodes': [4],
                'depth': 2,
                'rc': 4.0,
                'n_basis': 4,
                'atom_types': [1]}},
        'model': {
            'name': 'potential_model',
            'params': {'use_force': False}}}

    model = pinn.get_model(params)
    model.train(input_fn=lambda: load_numpy(data).repeat().shuffle(n).apply(
        sparse_batch(4)), max_steps=1)

    atoms = Atoms('H3', positions=[[0., 0., 0.], [0., 0., 0.9], [0., 0.9, 0.]])
    energies = {}
    for name in ('float32', 'float64'):
        calc = PiNN_calc(pinn.get_model(params), properties=['energy'],
                         default_dtype=name)
        atoms.calc = calc
        energies[name] = float(atoms.get_potential_energy())
        assert np.isfinite(energies[name])
    # same weights, only the arithmetic dtype differs
    assert energies['float32'] == pytest.approx(energies['float64'], abs=1e-3)
    apply_dtype('float32')

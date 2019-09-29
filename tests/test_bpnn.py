# -*- coding: utf-8 -*-
"""unit tests for bpnn implementation"""

import numpy as np
import tensorflow as tf
from helpers import assert_almost_equal


def _manual_sfs():
    lambd = 1.0
    zeta = 1.0
    eta = 0.01
    Rc = 12.0
    Rs = 0.5

    a = np.array([0., 0., 0.])
    b = np.array([1., 0., 0.])
    c = np.array([1., 1., 0.])
    ab = b-a
    ac = c-a
    bc = c-b
    Rab = np.linalg.norm(ab)
    Rac = np.linalg.norm(ac)
    Rbc = np.linalg.norm(bc)
    cosabc = np.dot(ab, ac)/(Rab*Rac)

    def fcut(R, Rcut):
        return 0.5*(np.cos(np.pi*R/Rcut)+1)
    abc = np.arccos(cosabc) * 180/np.pi

    g2_a = np.exp(-eta*(Rab-Rs))*fcut(Rab, Rc) +\
        np.exp(-eta*(Rac-Rs))*fcut(Rac, Rc)
    g3_a = 2**(1-zeta) *\
        (1+lambd*cosabc)**zeta*np.exp(-eta*(Rab**2+Rac**2+Rbc**2)) *\
        fcut(Rab, Rc)*fcut(Rac, Rc)*fcut(Rbc, Rc)
    g4_a = 2**(1-zeta) *\
        (1+lambd*cosabc)**zeta*np.exp(-eta*(Rab**2+Rac**2)) *\
        fcut(Rab, Rc)*fcut(Rac, Rc)

    return g2_a, g3_a, g4_a


def test_sfs():
    # test the BP symmetry functions against manual calculations
    # units in the original runner format is Bohr
    from helpers import get_trivial_runner_ds
    from pinn.networks import bpnn
    from pinn.io import sparse_batch

    bohr2ang = 0.5291772109
    dataset = get_trivial_runner_ds().apply(sparse_batch(1))
    sf_spec = [
        {'type': 'G2', 'i': 1, 'j': 'ALL',
         'eta': [0.01/(bohr2ang**2)], 'Rs': [0.5*bohr2ang]},
        {'type': 'G3', 'i': 1, 'j': 8, 'k': 1,
         'eta': [0.01/(bohr2ang**2)], 'lambd': [1.0], 'zeta': [1.0]},
        {'type': 'G4', 'i': 1, 'j': 8, 'k': 1,
         'eta': [0.01/(bohr2ang**2)], 'lambd': [1.0], 'zeta': [1.0]}
    ]
    nn_spec = {8: [35, 35], 1: [35, 35]}
    tensors = dataset.make_one_shot_iterator().get_next()
    tensors = bpnn(tensors,
                   sf_spec=sf_spec, nn_spec=nn_spec, rc=12*bohr2ang,
                   preprocess=True)
    with tf.Session() as sess:
        out = sess.run(tensors)
    print(out)
    g2_a, g3_a, g4_a = _manual_sfs()
    assert_almost_equal(out['fp_0'][0], g2_a)
    assert_almost_equal(out['fp_1'][0], g3_a)
    assert_almost_equal(out['fp_2'][0], g4_a)


def test_jacob_bpnn():
    """Check BPNN jacobian calculation"""
    from ase.collections import g2
    from pinn.networks import bpnn

    # Define the test case
    sf_spec = [
        {'type': 'G2', 'i': 1, 'j': 1, 'Rs': [1., 2.], 'eta': [0.1, 0.5]},
        {'type': 'G2', 'i': 8, 'j': 1, 'Rs': [1., 2.], 'eta': [0.1, 0.5]},
        {'type': 'G2', 'i': "ALL", 'j': "ALL",
            'Rs': [1., 2.], 'eta': [0.1, 0.5]},
        {'type': 'G2', 'i': "ALL", 'j': 1, 'Rs': [1.], 'eta': [0.01]},
        {'type': 'G3', 'i': 1, 'j': 8, 'lambd': [
            0.5, 1.], 'zeta': [1., 2.], 'eta': [0.1, 0.2]},
        {'type': 'G3', 'i': "ALL", 'j': 8, 'lambd': [
            0.5, 1.], 'zeta': [1., 2.], 'eta': [0.1, 0.2]},
        {'type': 'G4', 'i': 8, 'j': 8, 'lambd': [
            0.5, 1.], 'zeta': [1., 2.], 'eta': [0.1, 0.2]},
        {'type': 'G4', 'i': 8, 'j': 8, 'k': 1, 'lambd': [
            0.5, 1.], 'zeta': [1., 2.], 'eta': [0.1, 0.2]}
    ]
    nn_spec = {8: [32, 32], 1: [32, 32]}
    water = g2['H2O']
    water.set_cell([3.1, 3.1, 3.1])
    water.set_pbc(True)
    water = water.repeat([2, 2, 2])
    pos = water.get_positions()
    water.set_positions(pos+np.random.uniform(0, 0.2, pos.shape))

    tf.reset_default_graph()
    tensors = {
        "coord": tf.constant(water.positions, tf.float32),
        "ind_1": tf.zeros_like(water.numbers[:, np.newaxis], tf.int32),
        "elems": tf.constant(water.numbers, tf.int32),
        "cell":  tf.constant(water.cell[np.newaxis, :, :], tf.float32)
    }
    en = bpnn(tensors, sf_spec, nn_spec)
    frc = - tf.gradients(en, tensors['coord'])[0]
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        frc_jacob = sess.run(frc)
        saver.save(sess, '/tmp/test_jacob_bpnn.ckpt')

    tf.reset_default_graph()
    tensors = {
        "coord": tf.constant(water.positions, tf.float32),
        "ind_1": tf.zeros_like(water.numbers[:, np.newaxis], tf.int32),
        "elems": tf.constant(water.numbers, tf.int32),
        "cell":  tf.constant(water.cell[np.newaxis, :, :], tf.float32)
    }
    en = bpnn(tensors, sf_spec, nn_spec, use_jacobian=False)
    frc = - tf.gradients(en, tensors['coord'])[0]
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, '/tmp/test_jacob_bpnn.ckpt')
        frc_no_jacob = sess.run(frc)
    assert np.all(np.abs(frc_jacob - frc_no_jacob) < 1e-3)

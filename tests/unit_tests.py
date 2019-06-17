import tensorflow as tf
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

def test_jacob_pinn():
    """Check BPNN jacobian calculation"""
    from ase.collections import g2
    from pinn.networks import pinn_network

    water = g2['H2O']
    water.set_cell([3.1, 3.1, 3.1])
    water.set_pbc(True)
    water = water.repeat([2,2,2])
    pos = water.get_positions()
    water.set_positions(pos+np.random.uniform(0,0.2,pos.shape))
    
    tf.reset_default_graph()    
    tensors = {
        "coord": tf.constant(water.positions, tf.float32),
        "ind_1": tf.zeros_like(water.numbers[:, np.newaxis], tf.int32),
        "elems": tf.constant(water.numbers, tf.int32),
        "cell":  tf.constant(water.cell[np.newaxis,:,:], tf.float32)
    }
    en = pinn_network(tensors)
    frc = - tf.gradients(en, tensors['coord'])[0]
    saver = tf.train.Saver()    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        frc_jacob = sess.run(frc)
        print(frc_jacob)
        saver.save(sess, '/tmp/test_jacob_pinn.ckpt')
        
    tf.reset_default_graph()
    tensors = {
        "coord": tf.constant(water.positions, tf.float32),
        "ind_1": tf.zeros_like(water.numbers[:, np.newaxis], tf.int32),
        "elems": tf.constant(water.numbers, tf.int32),
        "cell":  tf.constant(water.cell[np.newaxis,:,:], tf.float32)
    }        
    en = pinn_network(tensors, use_jacobian=False)
    frc = - tf.gradients(en, tensors['coord'])[0]
    saver = tf.train.Saver()        
    with tf.Session() as sess:
        saver.restore(sess, '/tmp/test_jacob_pinn.ckpt')        
        frc_no_jacob = sess.run(frc)
    assert np.all(np.abs(frc_jacob - frc_no_jacob) < 1e-3)

def test_jacob_bpnn():
    """Check BPNN jacobian calculation"""
    from ase.collections import g2
    from pinn.networks import bpnn_network

    # Define the test case
    sf_spec = [
        {'type':'G2', 'i': 1, 'j': 1, 'Rs': [1., 2.], 'etta': [0.1, 0.5]},
        {'type':'G2', 'i': 8, 'j': 1, 'Rs': [1., 2.], 'etta': [0.1, 0.5]},
        {'type':'G2', 'i': "ALL", 'j': "ALL", 'Rs': [1., 2.], 'etta': [0.1, 0.5]},
        {'type':'G2', 'i': "ALL", 'j': 1, 'Rs': [1.], 'etta': [0.01]},
        {'type':'G4', 'i': 8, 'j': 8, 'lambd':[0.5,1.], 'zeta': [1.,2.], 'etta': [0.1,0.2]},
        {'type':'G4', 'i': 8, 'j': 8, 'k': 1, 'lambd':[0.5,1.], 'zeta': [1.,2.], 'etta': [0.1,0.2]}        
    ]
    nn_spec = {8: [32,32], 1: [32,32]}
    water = g2['H2O']
    water.set_cell([3.1, 3.1, 3.1])
    water.set_pbc(True)
    water = water.repeat([2,2,2])
    pos = water.get_positions()
    water.set_positions(pos+np.random.uniform(0,0.2,pos.shape))
    
    tf.reset_default_graph()    
    tensors = {
        "coord": tf.constant(water.positions, tf.float32),
        "ind_1": tf.zeros_like(water.numbers[:, np.newaxis], tf.int32),
        "elems": tf.constant(water.numbers, tf.int32),
        "cell":  tf.constant(water.cell[np.newaxis,:,:], tf.float32)
    }
    en = bpnn_network(tensors, sf_spec, nn_spec)
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
        "cell":  tf.constant(water.cell[np.newaxis,:,:], tf.float32)
    }        
    en = bpnn_network(tensors, sf_spec, nn_spec, use_jacobian=False)
    frc = - tf.gradients(en, tensors['coord'])[0]
    saver = tf.train.Saver()  
    with tf.Session() as sess:
        saver.restore(sess, '/tmp/test_jacob_bpnn.ckpt')        
        frc_no_jacob = sess.run(frc)
    assert np.all(np.abs(frc_jacob - frc_no_jacob) < 1e-3)

def test_derivitives():
    """ Test the calcualted derivitives: forces and stress
    with a LJ calculator against ASE's implementation
    """
    from ase.collections import g2
    from ase.build import bulk
    from ase.calculators.lj import LennardJones
    from pinn.models import potential_model
    from pinn.calculator import PiNN_calc
    import numpy as np
    params = {
    'model_dir': '',
    'network':'lj',
    'netparam': {'rc':3},
    'train':{}}
    model = potential_model(params)
    pi_lj = PiNN_calc(model)
    test_set = [bulk('Cu').repeat([3,3,3]), bulk('Mg'), g2['H2O']]
    for atoms in test_set:
        pos = atoms.get_positions()
        atoms.set_positions(pos+np.random.uniform(0,0.2,pos.shape))
        atoms.set_calculator(pi_lj)
        f_pinn, e_pinn = atoms.get_forces(), atoms.get_potential_energy()
        atoms.set_calculator(LennardJones())
        f_ase, e_ase = atoms.get_forces(), atoms.get_potential_energy()
        assert np.all(np.abs(f_pinn-f_ase)<1e-3)
        assert np.abs(e_pinn-e_ase)<1e-3
        if np.any(atoms.pbc):
            atoms.set_calculator(pi_lj)
            s_pinn = atoms.get_stress()
            atoms.set_calculator(LennardJones())
            s_ase = atoms.get_stress()
            assert np.all(np.abs(s_pinn-s_ase)<1e-3)

def test_clist_nl():
    """Cell list neighbor test
    Compare with ASE implementation
    """
    from ase.build import bulk
    from ase.neighborlist import neighbor_list
    from pinn.layers import cell_list_nl
    
    to_test = [bulk('Cu'), bulk('Mg'), bulk('Fe')]
    ind, coord, cell = [],[],[]
    for i, a in enumerate(to_test):
        ind.append([[i]]*len(a))
        coord.append(a.positions)
        cell.append(a.cell)

    with tf.Graph().as_default():
        tensors = {
            'ind_1': tf.constant(np.concatenate(ind, axis=0), tf.int32),
            'coord': tf.constant(np.concatenate(coord, axis=0), tf.float32),
            'cell': tf.constant(np.stack(cell, axis=0), tf.float32)}
        nl = cell_list_nl(tensors, rc=10)
        with tf.Session() as sess:
            dist_pinn = sess.run(nl['dist'])

    dist_ase = []
    for a in to_test:
        dist_ase.append(neighbor_list('d', a, 10))
    dist_ase = np.concatenate(dist_ase,0)
    assert np.all(np.sort(dist_ase)-np.sort(dist_pinn)<1e-4)
    


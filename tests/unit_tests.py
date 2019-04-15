import tensorflow as tf
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

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
        assert np.all((f_pinn-f_ase)<1e-3)
        assert (e_pinn-e_ase)<1e-4
        if np.any(atoms.pbc):
            atoms.set_calculator(pi_lj)
            s_pinn = atoms.get_stress()
            atoms.set_calculator(LennardJones())
            s_ase = atoms.get_stress()
            assert np.all((s_pinn-s_ase)<1e-4)


def test_clist_nl():
    """Cell list neighbor test
    Compare with ASE implementation
    """
    from ase.build import bulk
    import pinn.filters as f
    from ase.neighborlist import neighbor_list
    to_test = [bulk('Cu'), bulk('Mg'), bulk('Fe')]
    ind, coord, cell = [],[],[]
    for i, a in enumerate(to_test):
        ind.append([[i]]*len(a))
        coord.append(a.positions)
        cell.append(a.cell)

    with tf.Graph().as_default():
        tensors = {
            'ind': {1: tf.constant(np.concatenate(ind, axis=0), tf.int32)},
            'coord': tf.constant(np.concatenate(coord, axis=0), tf.float32),
            'cell': tf.constant(np.stack(cell, axis=0), tf.float32)}
        f.cell_list_nl(10)(tensors)
        with tf.Session() as sess:
            dist_pinn = sess.run(tensors['dist'])

    dist_ase = []
    for a in to_test:
        dist_ase.append(neighbor_list('d', a, 10))
    dist_ase = np.concatenate(dist_ase,0)
    assert np.all(np.sort(dist_ase)-np.sort(dist_pinn)<1e-4)
    


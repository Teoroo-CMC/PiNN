import tempfile, os
import tensorflow as tf
import numpy as np
from shutil import rmtree


def test_potential_model():
    """A simple example to test training and using a potential"""
    from ase import Atoms    
    from ase.calculators.lj import LennardJones
    from pinn.io import load_numpy, sparse_batch
    from pinn.models import potential_model
    def three_body_sample(atoms, a, r):
        x = a * np.pi / 180
        pos = [[0, 0, 0],
               [0, 2, 0],
               [0, r*np.cos(x), r*np.sin(x)]]
        atoms.set_positions(pos)
        return atoms
    
    tmp = tempfile.mkdtemp(prefix='pinn_test')
    atoms = Atoms('H3', calculator=LennardJones())
    na, nr = 50, 50
    arange = np.linspace(30,180,na)
    rrange = np.linspace(1,3,nr)
    # Truth
    agrid, rgrid = np.meshgrid(arange, rrange)
    egrid = np.zeros([na, nr])
    for i in range(na):
        for j in range(nr):
            atoms = three_body_sample(atoms, arange[i], rrange[j])
            egrid[i,j] = atoms.get_potential_energy()
    # Samples
    nsample = 50
    asample, rsample = [], []
    distsample = []
    data = {'e_data':[], 'f_data':[], 'elems':[], 'coord':[]}
    for i in range(nsample):
        a, r = np.random.choice(arange), np.random.choice(rrange)
        atoms = three_body_sample(atoms, a, r)
        dist = atoms.get_all_distances()
        dist = dist[np.nonzero(dist)]
        data['e_data'].append(atoms.get_potential_energy())
        data['f_data'].append(atoms.get_forces())
        data['coord'].append(atoms.get_positions())
        data['elems'].append(atoms.numbers)
        asample.append(a)
        rsample.append(r)
        distsample.append(dist)
    data = {k:np.array(v) for k,v in data.items()}
    dataset = lambda: load_numpy(data)
    train = lambda: dataset()['train'].shuffle(100).repeat().apply(sparse_batch(100))
    test = lambda: dataset()['test'].repeat().apply(sparse_batch(100))
    params={
    'model_dir': tmp,
    'network': 'pinn_network',
    'network_params': {
        'ii_nodes':[8,8],
        'pi_nodes':[8,8],
        'pp_nodes':[8,8],
        'en_nodes':[8,8],
        'rc': 3.0,
        'atom_types':[1]},
    'model_params':{
        'use_force': True}
    }
    model = potential_model(params)
    train_spec = tf.estimator.TrainSpec(input_fn=train, max_steps=200)
    eval_spec = tf.estimator.EvalSpec(input_fn=test, steps=10)
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    rmtree(tmp, ignore_errors=True)    


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
    'network_params': {'rc':3},
    'model_params':{}}
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
    


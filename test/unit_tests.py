import tensorflow as tf
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

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
    tensors = {
        'ind': {1: tf.constant(np.concatenate(ind, axis=0), tf.int32)},
        'coord': tf.constant(np.concatenate(coord, axis=0), tf.float32),
        'cell': tf.constant(np.stack(cell, axis=0), tf.float32)}
    f.cell_list_nl(10)(tensors)
    print(tensors)
    with tf.Session() as sess:
        dist_pinn = sess.run(tensors['dist'])

    dist_ase = []
    for a in to_test:
        dist_ase.append(neighbor_list('d', a, 10))
    dist_ase = np.concatenate(dist_ase,0)
    assert np.all(np.sort(dist_ase)-np.sort(dist_pinn)<1e-4)
    


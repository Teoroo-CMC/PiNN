# -*- coding: utf-8 -*-
"""This file implements different models to predict the charge response kernel (CRK)
and polarizability tensor by fitting polarizability tensor data.

All models output the polarizability tensor 'alpha' and CRK 'chi'. 

Implemented models: ACKS2, EEM, EtaInv, Local chi, and Local. 
For model details see ref. Shao, Y.; Andersson, L.; Knijff, L.; Zhang, C., Finite-field coupling via learning the charge response kernel. Electron. Struct. 2022, 4, 014012. 
The same models are also implemented with the addition of an atomic polarizability (isotropic) term. 
"""

from .pol_utils import *
from pinn import get_network

ang2bohr = 1.8897259886 # convert R-> bohr, all chi, alpha should be interpreted as a.u.

@export_pol_model('pol_acks2_model')
def pol_acks2_fn(tensors, params):
    params['network']['params'].update({'out_extra': {'i1':1,'i3':1}})
    network = get_network(params['network'])
    tensors = network.preprocess(tensors)
    ppred, ipred = network(tensors)
    ipred1 = ipred['i1']
    ipred3 = ipred['i3']
    i3norm = tf.einsum('aij,aij->aj',ipred3,ipred3)
    ipred = ipred1 + tf.einsum('aij,aij->aj',ipred3,ipred3)

    # construct eta_e, this part should e same with EEM
    atom_rind, pair_rind = make_indices(tensors)
    nmax = tf.reduce_max(atom_rind[:, 1])+1
    nbatch = tf.reduce_max(atom_rind[:,0])+1
    sigma_a, sigma_e = make_sigma(tensors['elems'], trainable=True)
    for i, e in enumerate(params['network']['params']['atom_types']):
        tf.compat.v1.summary.scalar(f'sigma/{e}', sigma_e[i])

    cell = tensors['cell'] if 'cell' in tensors else None
    ewald_params = {k: params['model']['params'][f'ewald_{k}'] for k in ['kmax', 'rc', 'eta']}
    E = make_E(atom_rind, tensors['coord'], sigma_a, nbatch, nmax, cell=cell,
               **ewald_params)/ang2bohr
    J = make_diag(atom_rind, tf.abs(ppred), nbatch, nmax)
    eta_e = E+J

    # construct chi_s
    chi_s = make_offdiag(pair_rind, tf.abs(ipred[:,0]), nbatch, nmax,
                         symmetric=True, invariant=True)
    I     = tf.eye(nmax, batch_shape=[nbatch])
    chi   = tf.linalg.solve(
        I-tf.linalg.einsum('bij,bjk->bik', eta_e, chi_s), chi_s, adjoint=True)

    R = make_R(atom_rind, tensors['coord'], nbatch, nmax)*ang2bohr
    alpha = - tf.linalg.einsum('bix,bij,bjy->bxy', R, chi, R)
    return {'alpha':alpha, 'chi':chi, 'eta_e': eta_e, 'chi_s': chi_s}


@export_pol_model('pol_eem_model')
def pol_eem_fn(tensors, params):
    #params['network']['params'].update({'ppred': 1, 'ipred': 0})
    network = get_network(params['network'])
    tensors = network.preprocess(tensors)
    ppred = network(tensors)

    # construct EEM, using trainale sigma
    atom_rind, _ = make_indices(tensors)
    nbatch = tf.reduce_max(atom_rind[:,0])+1
    nmax = tf.reduce_max(atom_rind[:, 1])+1
    sigma_a, sigma_e = make_sigma(tensors['elems'], trainable=True)
    for i, e in enumerate(params['network']['params']['atom_types']):
        tf.compat.v1.summary.scalar(f'sigma/{e}', sigma_e[i])

    cell = tensors['cell'] if 'cell' in tensors else None
    ewald_params = {k: params['model']['params'][f'ewald_{k}'] for k in ['kmax', 'rc', 'eta']}
    E = make_E(atom_rind, tensors['coord'], sigma_a, nbatch, nmax, cell=cell,
               **ewald_params)/ang2bohr
    J = make_diag(atom_rind, tf.abs(ppred), nbatch, nmax)
    D = make_dummy(atom_rind, nbatch, nmax)
    eta    = E+J
    R = make_R(atom_rind, tensors['coord'], nbatch, nmax)*ang2bohr
    etaInv = tf.linalg.inv(eta+D)-D
    egap = 1/tf.einsum('aij->a',etaInv)
    chi    = make_lrf(etaInv)

    alpha = -tf.linalg.einsum('bix,bij,bjy->bxy', R, chi, R)
    return {'alpha':alpha, 'egap': egap, 'chi':chi, 'eta': eta}


@export_pol_model('pol_etainv_model')
def pol_etainv_fn(tensors, params):
    from pinn import get_network
    eps = 0.01  #
    if 'epsilon' in params['model']['params']:
        eps = params['model']['params']['epsilon']

    params['network']['params'].update({'out_extra': {'i1':1, 'i3':1}})
    network = get_network(params['network'])
    tensors = network.preprocess(tensors)
    ppred, ipred = network(tensors)
    ipred1 = ipred['i1']
    ipred3 = ipred['i3']
    i3norm = tf.einsum('aij,aij->aj',ipred3,ipred3)
    ipred = ipred1 + tf.einsum('aij,aij->aj',ipred3,ipred3)
    atom_rind,pair_rind = make_indices(tensors)
    nbatch = tf.reduce_max(atom_rind[:,0])+1
    nmax = tf.reduce_max(atom_rind[:, 1])+1

    M = make_offdiag(pair_rind, ipred[:,0], nbatch, nmax,symmetric=False, invariant=False)
    M += make_diag(atom_rind, ppred, nbatch, nmax)
    etaInv = tf.einsum('aij,aik->ajk',M,M)
    etaInv += make_diag(atom_rind, tf.ones(tf.shape(atom_rind)[0]), nbatch, nmax)*eps
    egap = 1/tf.einsum('aij->a',etaInv)
    chi = make_lrf(etaInv)

    #chi -> alpha
    R = make_R(atom_rind, tensors['coord'], nbatch, nmax)*ang2bohr
    alpha = -tf.linalg.einsum('bix,bij,bjy->bxy', R, chi, R)
    return {'alpha':alpha, 'egap': egap, 'chi': chi}


@export_pol_model('pol_localchi_model')
def pol_localchi_fn(tensors, params):
    """ Similar to acks2 model, but construct chi directly instead of chi_s
    """
    params['network']['params'].update({'out_extra': {'i1':1,'i3':1}})
    network = get_network(params['network'])
    tensors = network.preprocess(tensors)
    ppred, ipred = network(tensors)
    ipred1 = ipred['i1']
    ipred3 = ipred['i3']
    i3norm = tf.einsum('aij,aij->aj',ipred3,ipred3)
    ipred = ipred1 + tf.einsum('aij,aij->aj',ipred3,ipred3)
    atom_rind,pair_rind = make_indices(tensors)
    nbatch = tf.reduce_max(atom_rind[:,0])+1
    nmax = tf.reduce_max(atom_rind[:, 1])+1

    # chi -> alpha
    chi = make_offdiag(pair_rind, tf.abs(ipred[:,0]), nbatch, nmax,
                       symmetric=True, invariant=True)

    R = make_R(atom_rind, tensors['coord'], nbatch, nmax)*ang2bohr
    alpha = -tf.linalg.einsum('bix,bij,bjy->bxy', R, chi, R)
    return {'alpha':alpha, 'chi':chi}


@export_pol_model('pol_local_model')
def pol_local_fn(tensors, params):
    from tensorflow.math import unsorted_segment_sum
    def _form_triplet(tensors):
        """Returns triplet indices [ij, jk], where r_ij, r_jk < r_c"""
        p_iind = tensors['ind_2'][:, 0]
        n_atoms = tf.shape(tensors['ind_1'])[0]
        n_pairs = tf.shape(tensors['ind_2'])[0]
        p_aind = tf.cumsum(tf.ones(n_pairs, tf.int32))
        p_rind = p_aind - tf.gather(tf.math.segment_min(p_aind, p_iind), p_iind)
        t_dense = tf.scatter_nd(tf.stack([p_iind, p_rind], axis=1), p_aind,
                                [n_atoms, tf.reduce_max(p_rind)+1])
        t_dense = tf.gather(t_dense, p_iind)
        t_index = tf.cast(tf.where(t_dense), tf.int32)
        t_ijind = t_index[:, 0]
        t_ikind = tf.gather_nd(t_dense, t_index)-1
        t_ind = tf.stack([t_ijind, t_ikind], axis=1)
        return t_ind

    params['network']['params'].update({'out_extra': {'i1':1,'i3':1}})
    network = get_network(params['network'])
    tensors = network.preprocess(tensors)
    ppred, ipred = network(tensors)
    ipred1 = ipred['i1']
    ipred3 = ipred['i3']
    i3norm = tf.einsum('aij,aij->aj',ipred3,ipred3)
    ipred = ipred1 + tf.einsum('aij,aij->aj',ipred3,ipred3)
    ind1 = tensors['ind_1'][:,0]
    natoms = tf.shape(ind1)[0]
    nbatch = tf.reduce_max(ind1)+1
    ind2 = tensors['ind_2']
    diff = tensors['diff']*ang2bohr
    # tmp -> n_atoms x n_pred x 3 -> local "basis" for polarizability
    tmp = unsorted_segment_sum(tf.einsum('pc,px->pcx', ipred, diff), ind2[:,0], natoms)
    alpha_i = tf.einsum('pcx,pcy->pxy', tmp, tmp)
    alpha = unsorted_segment_sum(alpha_i, ind1, nbatch)
    # chi is a bit more expensive to get :(
    aind = tf.cumsum(tf.ones_like(ind1))     # "absolute" pos of atom in batch
    amax = tf.shape(ind1)[0]
    rind = aind-tf.gather(tf.math.unsorted_segment_min(aind, ind1, nbatch), ind1)
    nmax = tf.reduce_max(rind)+1
    ind2_b = tf.gather(ind1, ind2[:,0])
    ind2_i = tf.gather(rind, ind2[:,0])
    ind2_j = tf.gather(rind, ind2[:,1])
    ind3 = _form_triplet(tensors)
    # this returns [N_tri, 2] array with positions of [idx_ij, idx_ik] in idx2
    ind3_b = tf.gather(ind2_b, ind3[:,0])
    ind3_i = tf.gather(ind2_i, ind3[:,0])
    ind3_j = tf.gather(ind2_j, ind3[:,0])
    ind3_k = tf.gather(ind2_j, ind3[:,1])
    y_ij  = tf.gather(ipred, ind3[:,0])
    y_ik  = tf.gather(ipred, ind3[:,1])
    y_ijk = tf.einsum('tc,tc->t', y_ij, y_ik)
    i_bjk = tf.stack([ind3_b, ind3_j, ind3_k], axis=1)
    i_bji = tf.stack([ind3_b, ind3_j, ind3_i], axis=1)
    i_bik = tf.stack([ind3_b, ind3_i, ind3_k], axis=1)
    i_bii = tf.stack([ind3_b, ind3_i, ind3_i], axis=1)
    shape = [nbatch, nmax, nmax]
    chi = tf.zeros(shape)\
        - tf.scatter_nd(i_bjk, y_ijk, shape)\
        + tf.scatter_nd(i_bji, y_ijk, shape)\
        + tf.scatter_nd(i_bik, y_ijk, shape)\
        - tf.scatter_nd(i_bii, y_ijk, shape)
    return {'alpha': alpha, 'chi': chi}

@export_pol_model('pol_acks2_iso_model')
def pol_acks2_iso_fn(tensors, params):
    params['network']['params'].update({'out_extra': {'p1':1,'i1':1,'i3':1}})
    network = get_network(params['network'])
    tensors = network.preprocess(tensors)
    ppred, ipred = network(tensors)
    ipred1 = ipred['i1']
    ipred3 = ipred['i3']
    i3norm = tf.einsum('aij,aij->aj',ipred3,ipred3)
    p12 = ipred['p1']
    ipred = ipred1 + tf.einsum('aij,aij->aj',ipred3,ipred3)

    # construct eta_e, this part should e same with EEM
    atom_rind, pair_rind = make_indices(tensors)
    nmax = tf.reduce_max(atom_rind[:, 1])+1
    nbatch = tf.reduce_max(atom_rind[:,0])+1
    sigma_a, sigma_e = make_sigma(tensors['elems'], trainable=True)
    for i, e in enumerate(params['network']['params']['atom_types']):
        tf.compat.v1.summary.scalar(f'sigma/{e}', sigma_e[i])

    cell = tensors['cell'] if 'cell' in tensors else None
    ewald_params = {k: params['model']['params'][f'ewald_{k}'] for k in ['kmax', 'rc', 'eta']}
    E = make_E(atom_rind, tensors['coord'], sigma_a, nbatch, nmax, cell=cell,
               **ewald_params)/ang2bohr
    J = make_diag(atom_rind, tf.abs(ppred), nbatch, nmax)
    eta_e = E+J

    # construct chi_s
    chi_s = make_offdiag(pair_rind, tf.abs(ipred[:,0]), nbatch, nmax,
                         symmetric=True, invariant=True)
    I     = tf.eye(nmax, batch_shape=[nbatch])
    chi   = tf.linalg.solve(
        I-tf.linalg.einsum('bij,bjk->bik', eta_e, chi_s), chi_s, adjoint=True)

    R = make_R(atom_rind, tensors['coord'], nbatch, nmax)*ang2bohr
    alpha = - tf.linalg.einsum('bix,bij,bjy->bxy', R, chi, R)
    alpha_iso = tf.eye(3,batch_shape=[nbatch])*tf.math.unsorted_segment_sum(p12,tensors['ind_1'],nbatch)[:,None,None]
    alpha += alpha_iso
    return {'alpha':alpha, 'alpha_iso':alpha_iso, 'chi':chi, 'eta_e': eta_e, 'chi_s': chi_s}

@export_pol_model('pol_eem_iso_model')
def pol_eem_iso_fn(tensors, params):
    params['network']['params'].update({'out_extra': {'p1':1}})
    network = get_network(params['network'])
    tensors = network.preprocess(tensors)
    ppred,ipred = network(tensors)
    p12 = ipred['p1']

    # construct EEM, using trainale sigma
    atom_rind, _ = make_indices(tensors)
    nbatch = tf.reduce_max(atom_rind[:,0])+1
    nmax = tf.reduce_max(atom_rind[:, 1])+1
    sigma_a, sigma_e = make_sigma(tensors['elems'], trainable=True)
    for i, e in enumerate(params['network']['params']['atom_types']):
        tf.compat.v1.summary.scalar(f'sigma/{e}', sigma_e[i])

    cell = tensors['cell'] if 'cell' in tensors else None
    ewald_params = {k: params['model']['params'][f'ewald_{k}'] for k in ['kmax', 'rc', 'eta']}
    E = make_E(atom_rind, tensors['coord'], sigma_a, nbatch, nmax, cell=cell,
               **ewald_params)/ang2bohr
    J = make_diag(atom_rind, tf.abs(ppred), nbatch, nmax)
    D = make_dummy(atom_rind, nbatch, nmax)
    eta    = E+J
    etaInv = tf.linalg.inv(eta+D)-D
    egap = 1/tf.einsum('aij->a',etaInv)
    chi    = make_lrf(etaInv)

    R = make_R(atom_rind, tensors['coord'], nbatch, nmax)*ang2bohr

    alpha = -tf.linalg.einsum('bix,bij,bjy->bxy', R, chi, R)
    alpha_iso = tf.eye(3,batch_shape=[nbatch])*tf.math.unsorted_segment_sum(p12,tensors['ind_1'],nbatch)[:,None,None]
    alpha += alpha_iso
    return {'alpha':alpha, 'alpha_iso':alpha_iso, 'egap': egap, 'chi':chi, 'eta': eta}

@export_pol_model('pol_etainv_iso_model')
def pol_etainv_iso_fn(tensors, params):
    from pinn import get_network
    eps = 0.01  #
    if 'epsilon' in params['model']['params']:
        eps = params['model']['params']['epsilon']

    params['network']['params'].update({'out_extra': {'p1':1,'i1':1, 'i3':1}})
    network = get_network(params['network'])
    tensors = network.preprocess(tensors)
    ppred, ipred = network(tensors)
    p12 = ipred['p1']
    ipred1 = ipred['i1']
    ipred3 = ipred['i3']
    i3norm = tf.einsum('aij,aij->aj',ipred3,ipred3)
    ipred = ipred1 + tf.einsum('aij,aij->aj',ipred3,ipred3)
    atom_rind,pair_rind = make_indices(tensors)
    nbatch = tf.reduce_max(atom_rind[:,0])+1
    nmax = tf.reduce_max(atom_rind[:, 1])+1

    M = make_offdiag(pair_rind, ipred[:,0], nbatch, nmax,symmetric=False, invariant=False)
    M += make_diag(atom_rind, ppred, nbatch, nmax)
    etaInv = tf.einsum('aij,aik->ajk',M,M)
    etaInv += make_diag(atom_rind, tf.ones(tf.shape(atom_rind)[0]), nbatch, nmax)*eps
    egap = 1/tf.einsum('aij->a',etaInv)
    chi = make_lrf(etaInv)

    #chi -> alpha
    R = make_R(atom_rind, tensors['coord'], nbatch, nmax)*ang2bohr
    alpha = -tf.linalg.einsum('bix,bij,bjy->bxy', R, chi, R)
    alpha_iso = tf.eye(3,batch_shape=[nbatch])*tf.math.unsorted_segment_sum(p12,tensors['ind_1'],nbatch)[:,None,None]
    alpha += alpha_iso
    return {'alpha':alpha, 'alpha_iso':alpha_iso, 'egap': egap, 'chi': chi, 'M': M}

@export_pol_model('pol_localchi_iso_model')
def pol_localchi_iso_fn(tensors, params):
    """ Similar to acks2 model, but construct chi directly instead of chi_s
    """
    params['network']['params'].update({'out_extra': {'i1':1,'i3':1}})
    network = get_network(params['network'])
    tensors = network.preprocess(tensors)
    ppred, ipred = network(tensors)
    ipred1 = ipred['i1']
    ipred3 = ipred['i3']
    i3norm = tf.einsum('aij,aij->aj',ipred3,ipred3)
    ipred = ipred1 + tf.einsum('aij,aij->aj',ipred3,ipred3)
    atom_rind,pair_rind = make_indices(tensors)
    nbatch = tf.reduce_max(atom_rind[:,0])+1
    nmax = tf.reduce_max(atom_rind[:, 1])+1

    # chi -> alpha
    chi = make_offdiag(pair_rind, tf.abs(ipred[:,0]), nbatch, nmax,
                       symmetric=True, invariant=True)

    R = make_R(atom_rind, tensors['coord'], nbatch, nmax)*ang2bohr
    alpha = -tf.linalg.einsum('bix,bij,bjy->bxy', R, chi, R)
    alpha_iso = tf.eye(3,batch_shape=[nbatch])*tf.math.unsorted_segment_sum(ppred[:,None],tensors['ind_1'],nbatch)[:,None,None]
    alpha += alpha_iso
    return {'alpha':alpha, 'alpha_iso':alpha_iso, 'chi':chi}

@export_pol_model('pol_local_iso_model')
def pol_local_iso_fn(tensors, params):
    from tensorflow.math import unsorted_segment_sum
    def _form_triplet(tensors):
        """Returns triplet indices [ij, jk], where r_ij, r_jk < r_c"""
        p_iind = tensors['ind_2'][:, 0]
        n_atoms = tf.shape(tensors['ind_1'])[0]
        n_pairs = tf.shape(tensors['ind_2'])[0]
        p_aind = tf.cumsum(tf.ones(n_pairs, tf.int32))
        p_rind = p_aind - tf.gather(tf.math.segment_min(p_aind, p_iind), p_iind)
        t_dense = tf.scatter_nd(tf.stack([p_iind, p_rind], axis=1), p_aind,
                                [n_atoms, tf.reduce_max(p_rind)+1])
        t_dense = tf.gather(t_dense, p_iind)
        t_index = tf.cast(tf.where(t_dense), tf.int32)
        t_ijind = t_index[:, 0]
        t_ikind = tf.gather_nd(t_dense, t_index)-1
        t_ind = tf.stack([t_ijind, t_ikind], axis=1)
        return t_ind

    params['network']['params'].update({'out_extra': {'i1':1,'i3':1}})
    network = get_network(params['network'])
    tensors = network.preprocess(tensors)
    ppred, ipred = network(tensors)
    ipred1 = ipred['i1']
    ipred3 = ipred['i3']
    i3norm = tf.einsum('aij,aij->aj',ipred3,ipred3)
    ipred = ipred1 + tf.einsum('aij,aij->aj',ipred3,ipred3)
    ind1 = tensors['ind_1'][:,0]
    natoms = tf.shape(ind1)[0]
    nbatch = tf.reduce_max(ind1)+1
    ind2 = tensors['ind_2']
    diff = tensors['diff']*ang2bohr
    # tmp -> n_atoms x n_pred x 3 -> local "basis" for polarizability
    tmp = unsorted_segment_sum(tf.einsum('pc,px->pcx', ipred, diff), ind2[:,0], natoms)
    alpha_i = tf.einsum('pcx,pcy->pxy', tmp, tmp)
    alpha = unsorted_segment_sum(alpha_i, ind1, nbatch)
    alpha_iso = tf.eye(3,batch_shape=[nbatch])*tf.math.unsorted_segment_sum(ppred,ind1,nbatch)[:,None,None]
    alpha += alpha_iso
    # chi is a bit more expensive to get :(
    aind = tf.cumsum(tf.ones_like(ind1))     # "absolute" pos of atom in batch
    amax = tf.shape(ind1)[0]
    rind = aind-tf.gather(tf.math.unsorted_segment_min(aind, ind1, nbatch), ind1)
    nmax = tf.reduce_max(rind)+1
    ind2_b = tf.gather(ind1, ind2[:,0])
    ind2_i = tf.gather(rind, ind2[:,0])
    ind2_j = tf.gather(rind, ind2[:,1])
    ind3 = _form_triplet(tensors)
    # this returns [N_tri, 2] array with positions of [idx_ij, idx_ik] in idx2
    ind3_b = tf.gather(ind2_b, ind3[:,0])
    ind3_i = tf.gather(ind2_i, ind3[:,0])
    ind3_j = tf.gather(ind2_j, ind3[:,0])
    ind3_k = tf.gather(ind2_j, ind3[:,1])
    y_ij  = tf.gather(ipred, ind3[:,0])
    y_ik  = tf.gather(ipred, ind3[:,1])
    y_ijk = tf.einsum('tc,tc->t', y_ij, y_ik)
    i_bjk = tf.stack([ind3_b, ind3_j, ind3_k], axis=1)
    i_bji = tf.stack([ind3_b, ind3_j, ind3_i], axis=1)
    i_bik = tf.stack([ind3_b, ind3_i, ind3_k], axis=1)
    i_bii = tf.stack([ind3_b, ind3_i, ind3_i], axis=1)
    shape = [nbatch, nmax, nmax]
    chi = tf.zeros(shape)\
        - tf.scatter_nd(i_bjk, y_ijk, shape)\
        + tf.scatter_nd(i_bji, y_ijk, shape)\
        + tf.scatter_nd(i_bik, y_ijk, shape)\
        - tf.scatter_nd(i_bii, y_ijk, shape)
    return {'alpha': alpha, 'alpha_iso': alpha_iso, 'chi': chi}
# -*- coding: utf-8 -*-
"""This file implements implements helper functions for pol_models. The code is updated for PiNet2 from the original PiNet-chi implementation by Yunqi Shao.
"""

import os, pinn, warnings
import numpy as np
import tensorflow as tf
import pinn.networks
import matplotlib.pyplot as plt
from pinn import get_network
from pinn.utils import pi_named
from pinn.models.base import export_model, get_train_op, MetricsCollector
index_warning = 'Converting sparse IndexedSlices'
deprecate_warning = 'deprecated'
warnings.filterwarnings('ignore', index_warning)
warnings.filterwarnings('ignore', deprecate_warning)

default_params = {
    'p_scale': 1.0,          # P scale for training
    'p_unit': 1.0,           # output unit of P during prediction
    'p_loss_multiplier': 1.0,# 
    'train_egap': 0,
    'eval_egap' : 0,
    # Ewald Sum Parameters, used in EEM & ACKS2, disabled by default
    'ewald_rc':   None,
    'ewald_kmax': None,
    'ewald_eta' : None,
}

implemented_models = {}

@pi_named("METRICS")
def make_metrics(features, predictions, params, mode):
    from pinn.utils import count_atoms
    metrics = MetricsCollector(mode)
    p_pred = predictions['alpha']
    p_data = features['ptensor']
    natoms = count_atoms(features['ind_1'], dtype=p_data.dtype)[:,None,None]
    p_pred /= natoms
    p_data /= natoms
    diag    = lambda x: tf.reduce_sum(tf.stack([x[:,i,i] for i in [0,1,2]]), axis=0)/np.sqrt(3)
    offdiag = lambda x: tf.stack([x[:,i,j] for i,j in zip([0,0,1],[1,2,2])]
                                 +[(2*x[:,2,2]-x[:,0,0]-x[:,1,1])/2/np.sqrt(3)]
                                 +[(  x[:,0,0]-x[:,1,1])/2])*np.sqrt(2)

    perror = p_pred - p_data
    if params['eval_egap'] == 1: 
        egap_data = features['egap']
        egap_pred = predictions['egap']   
        eerror = egap_pred - egap_data
    if params['train_egap'] == 1:
        metrics.add_error('egap', egap_pred, egap_data, log_error=False)
    else:
        metrics.add_error('alpha_per_atom', p_pred, p_data, log_error=False)
    if mode== tf.estimator.ModeKeys.TRAIN:
        tf.compat.v1.summary.scalar('alpha_per_atom_RMSE',
                                    tf.sqrt(tf.reduce_mean(perror**2)*9))
        tf.compat.v1.summary.scalar('alpha_per_atom_diag_RMSE',
                                    tf.sqrt(tf.reduce_mean(diag(perror)**2)))
        tf.compat.v1.summary.scalar('alpha_per_atom_offdiag_RMSE',
                                    tf.sqrt(tf.reduce_mean(offdiag(perror)**2)*5))
        if params['eval_egap'] == 1:
            tf.compat.v1.summary.scalar('egap_RMSE',tf.sqrt(tf.reduce_mean(eerror**2)))
                                    
    if mode== tf.estimator.ModeKeys.EVAL:
        metrics.METRICS['METRICS/alpha_per_atom_RMSE'] =\
            tf.compat.v1.metrics.root_mean_squared_error(p_pred*3, p_data*3)
        metrics.METRICS['METRICS/alpha_per_atom_diag_RMSE'] =\
            tf.compat.v1.metrics.root_mean_squared_error(diag(p_pred), diag(p_data))
        metrics.METRICS['METRICS/alpha_per_atom_offdiag_RMSE'] =\
            tf.compat.v1.metrics.root_mean_squared_error(offdiag(p_pred)*np.sqrt(5),
                                                         offdiag(p_data)*np.sqrt(5))
        if params['eval_egap'] == 1:
            metrics.METRICS['METRICS/egap_RMSE'] =\
                tf.compat.v1.metrics.root_mean_squared_error(egap_pred,egap_data)
    return metrics


def export_pol_model(name):
    def decorator(pol_fn):
        @export_model
        def model_fn(features, labels, mode, params):
            model_params = default_params.copy()
            model_params.update(params['model']['params'])
            params['model']['params'] = model_params
            predictions = pol_fn(features, params)
            tvars = tf.compat.v1.trainable_variables()

            if mode == tf.estimator.ModeKeys.TRAIN:
                metrics = make_metrics(features, predictions, model_params, mode)
                train_op = get_train_op(params['optimizer'], metrics, tvars)
                return tf.estimator.EstimatorSpec(
                    mode, loss=tf.reduce_sum(metrics.LOSS), train_op=train_op)

            if mode == tf.estimator.ModeKeys.EVAL:
                metrics = make_metrics(features, predictions, model_params, mode)
                return tf.estimator.EstimatorSpec(
                    mode, loss=tf.reduce_sum(metrics.LOSS),
                    eval_metric_ops=metrics.METRICS)

            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        implemented_models.update({name: model_fn})
    return decorator

# this is perhaps a better way to handle externel models, to be implemented in PiNN
def get_model(model_spec, **kwargs):
    import yaml, os
    import numpy as np
    import tensorflow as tf
    from tensorflow.python.lib.io.file_io import FileIO
    from datetime import datetime
    from pinn.models.potential import potential_model
    from pinn.models.dipole import dipole_model
    if isinstance(model_spec, str):
        if tf.io.gfile.exists('{}/params.yml'.format(model_spec)):
            params_file = os.path.join(model_spec, 'params.yml')
            with FileIO(params_file, 'r') as f:
                model_spec = dict(yaml.load(f, Loader=yaml.Loader),
                                  model_dir=model_spec)
        elif tf.io.gfile.exists(model_spec):
            params_file = model_spec
            with FileIO(params_file, 'r') as f:
                model_spec = yaml.load(f, Loader=yaml.Loader)
        else:
            raise ValueError(f'{model_spec} does not seem to be a parameter file or model_dir')
    else:
        model_dir = model_spec['model_dir']
        yaml.Dumper.ignore_aliases = lambda *args: True
        to_write = yaml.dump(model_spec)
        params_file = os.path.join(model_dir, 'params.yml')
        if not tf.io.gfile.isdir(model_dir):
            tf.io.gfile.makedirs(model_dir)
        if tf.io.gfile.exists(params_file):
            original = FileIO(params_file, 'r').read()
            if original != to_write:
                tf.io.gfile.rename(params_file, params_file+'.' +
                                   datetime.now().strftime('%y%m%d%H%M'))
        FileIO(params_file, 'w').write(to_write)
    model = implemented_models[model_spec['model']['name']](model_spec, **kwargs)
    return model
pinn.get_model = get_model


def make_indices(tensors):
    ind1 = tensors['ind_1']
    ind2 = tensors['ind_2']
    coord = tensors['coord']
    natoms = tf.shape(ind1)[0]
    nbatch = tf.reduce_max(ind1)+1
    aind = tf.cumsum(tf.ones_like(ind1))
    amax = tf.shape(ind1)[0]
    rind = aind-tf.gather(tf.math.unsorted_segment_min(aind, ind1, nbatch), ind1)
    rmax = tf.math.unsorted_segment_max(rind, ind1, nbatch)
    ind4J = tf.transpose([ind1, rind, rind])[0]

    atom_rind = tf.transpose([ind1, rind])[0]
    pair_rind = tf.transpose([
        tf.gather(ind1, ind2[:,0]),
        tf.gather(rind, ind2[:,0]),
        tf.gather(rind, ind2[:,1])])[0]
    return atom_rind, pair_rind


def make_sigma(elems,
               types=[1,6,7,8,16,17],
               sigma={1:0.312, 6:0.730, 7:0.709, 8:0.661, 16:1.048, 17:1.016},
               trainable=False):
    """ Construct a element-specifc sigma array

    Sigma will be positive by construction.

    Note: the sigma values shall have the same unit as the coordinates,
          which is defaulted to Angstrom in PiNN.

    Args:
       elems (tensor): A [N,] tensor containing the elements for each atom
       types (list): element present in the dataset
       sigma (dict): dictionary mapping the element to the
       trainable (bool): whether the sigma values are trainable

    Returns:
        sigma_a: A [N,] tensor containing the sigma value for each atom
        sigma_e: A [Nelems,] tensor conatining the sigma value for each element
    """
    from pinn.layers import AtomicOnehot
    indices = tf.where(AtomicOnehot(types)(elems))[:,1]
    sigma_e = tf.Variable([sigma[k] for k in types], trainable=trainable)
    sigma_e = tf.abs(sigma_e)
    return tf.gather(sigma_e, indices), sigma_e


def make_R(atom_rind, coord, nbatch, nmax):
    """ Cast the coordinates in to the dense form [nbatch, nmax, 3]"""
    nbatch = tf.reduce_max(atom_rind[:,0])+1
    nmax = nmax if nmax is not None else tf.reduce_max(atom_rind[:, 1])+1
    R = tf.scatter_nd(atom_rind, coord, [nbatch, nmax, 3])
    return R


def make_E(atom_rind, coord, sigma, nbatch, nmax, cell=None,
           kmax=None, eta=None, rc=None):
    """Construct the electrostatic interaction matrix

    For aperiodic systems:

    E_ij = | 1/(sigma_i sqrt(pi))              ;  for i==j
           | erf(r_ij/(sqrt(2) gamma_ij))/r_ij ;  for i!=j
           , where gamma_ij = sqrt(sigma_i^2+sigma_j^2).

    For periodic systems (only 3D periodicity supported):

    When `cell` is supplied, and any of the Ewald sum parameters is missing, the
    same formular is used with r_ij being computed with the minimum-image
    convention. Note that the minimum image convention works only for
    orthorhombic cell for now.

    When all Ewald sum parameters (kmax, eta and rc) are supplied, E_ij is
    computed with Ewald Summation:
    E = ES + EL + ESELF, where:
    ES_ij = erfc{r_ij/[sqrt(2)eta]}/r_ij-erfc{r_ij/[sqrt(2)gamma_ij]}/r_ij;
    EL_ij = 4pi*exp(-k^2eta^2/2)[cos(kr_i)cos(kr_j)+sin(kr_i)sin(kr_j)]/(V*k^2);
    ESELF_ii = 1/[sqrt(pi)sigma_i]-sqrt(2)/[sqrt(pi)eta].

    Args:
        atom_rind (tensor): a [N, 2] tensor containing the relative indices
        coord (tensor): [N, 3] coordinates
        sigma (tensor): [N] tensor
        nmax  (tensor): [] scalar tensor for the max no. of atoms in a structure
        cell  (tensor): (optional) [N, 3, 3] cell vectors
        rc     (float): (optional) cutoff for real-space summation
        eta    (float): (optional) width of auxillery Gaussian in Ewald sum
        kmax     (int): (optional) maximum k-vectors unsed in Ewald summ

    Returns:
        The E matrix with the shape [nbatch, nmax, nmax]

    """
    from itertools import product
    from numpy import pi
    from pinn.layers import CellListNL
    from tensorflow.math import erfc, erf

    # Compute E with Ewald sum, only perform when all arguments are available
    if not ((cell is None) or (kmax is None) or (eta is None) or (rc is None)):
        # ES: erfc{r_ij/[sqrt(2)eta]}/r_ij-erfc{r_ij/[sqrt(2)gamma_ij]}/r_ij
        cell_list = CellListNL(rc=rc) # (re)build the cell list from scratch
        ind_1 = atom_rind[:,:1]
        nl = cell_list({'ind_1': ind_1, 'coord':coord, 'cell': cell})
        rnorm, ind_2 = nl['dist'], nl['ind_2']
        _, pair_rind = make_indices({'ind_1':ind_1, 'ind_2':ind_2, 'coord':coord})
        etafac = 1./(np.sqrt(2)*eta)
        sigfac = 1/tf.sqrt(2*tf.reduce_sum(tf.gather(sigma**2,nl['ind_2']),axis=1))
        ES = tf.scatter_nd(pair_rind, (erfc(rnorm*etafac)-erfc(rnorm*sigfac))/rnorm,
                           [nbatch, nmax, nmax])
        # EL: exp(-k^2eta^2/2)(cos(kr_i)cos(kr_j)+sin(kr_i)sin(kr_j))/k^2, k!=0
        kvects = [np.arange(-kmax,kmax+1, dtype='float32') for i in range(3)]
        kvects = tf.stack([kvec for kvec in product(*kvects) if kvec!=(0,0,0)])
        kvects = 2*pi*tf.einsum('bxc,kc->bkx', tf.linalg.inv(cell), kvects)
        knorm = tf.linalg.norm(kvects, axis=-1)
        kconst = (1./knorm**2)*tf.exp(-knorm**2*eta**2/2)
        R = tf.fill([nbatch, nmax, 3], np.nan) # fill with nan, remove at end
        R = tf.tensor_scatter_nd_update(R, atom_rind, coord)
        kr = tf.einsum('bkx, bix->bki', kvects, R) # aka structure factor
        cssn = tf.stack([tf.cos(kr),tf.sin(kr)], axis=0)
        EL = 4*pi*tf.einsum('bk,sbki,sbkj->bij', kconst, cssn, cssn)
        V  = tf.linalg.det(cell)[:,None,None]
        EL = tf.where(tf.math.is_nan(EL), tf.zeros_like(EL), EL)/V
        # ESELF: 1/[sqrt(pi)sigma_i]-sqrt(2)/[sqrt(pi)eta]
        ESELF = -np.sqrt(2/pi)/eta*tf.eye(nmax, batch_shape=[nbatch])
        ESELF +=  tf.linalg.diag(
            tf.scatter_nd(atom_rind, np.sqrt(1./pi)/sigma, [nbatch, nmax]))
        E = ES + EL + ESELF
        return E

    # Compute E naitvely (N^2)
    R = tf.fill([nbatch, nmax, 3], np.nan)
    R = tf.tensor_scatter_nd_update(R, atom_rind, coord)
    R_ij = R[:,None,:,:] - R[:,:,None,:]
    if cell is not None:
        # Apply minimum image convention if cell is supplied.
        # Index names: b->batch, x->cartesian coord, c->cell vector, p->pair.
        R_ij_flat = tf.reshape(tf.transpose(R_ij, [0,3,1,2]), [nbatch,3,-1])
        cell_inv  = tf.linalg.inv(cell)
        R_ij_frac = tf.einsum('bxc,bxp->bcp', cell_inv, R_ij_flat)
        R_ij_frac -= tf.math.rint(R_ij_frac)
        R_ij_flat = tf.einsum('bcx,bcp->bxp', cell, R_ij_frac)
        R_ij = tf.transpose(tf.reshape(R_ij_flat,[nbatch,3,nmax,nmax]),[0,2,3,1])
    r_ij = tf.norm(R_ij, axis=3)
    r_ij = tf.where(tf.math.is_nan(r_ij), tf.zeros_like(r_ij), r_ij)
    sigma2 = tf.scatter_nd(atom_rind, sigma**2 , [nbatch, nmax])
    gamma_ij = tf.sqrt(sigma2[:,None,:]+sigma2[:,:,None])
    E = tf.math.divide_no_nan(erf(r_ij/gamma_ij/tf.sqrt(2.)), r_ij)
    E_diag = tf.scatter_nd(atom_rind, 1/tf.sqrt(pi)/sigma, [nbatch, nmax])
    return tf.linalg.set_diag(E, E_diag)


def make_lrf(Ainv, eps=1e-15):
    """ Construct the linear response function from the softness kernel

    Args:
        eta_inv (tensor): a [B, N, N] tensor containing the softness kernel
        eps (tensor): a small number

    Returns:
        chi (tensor): a [B, N, N] tensor containing the linear response kernel
    """
    Ad   = tf.einsum('bij->bi',Ainv)
    AddA = tf.einsum('bi,bj->bij', Ad, Ad)
    dAd  = tf.einsum('bi->b', Ad)+eps
    chi  = - Ainv + AddA/dAd[:,None,None]
    return chi

def make_diag(atom_rind, params, nbatch, nmax):
    """ Construct a diagonal matrix with atomistic predictions"""
    indices = tf.stack([atom_rind[:,i] for i in [0,1,1]], axis=1)
    return tf.scatter_nd(indices, params, [nbatch, nmax, nmax])

def make_offdiag(pair_rind, params, nbatch, nmax,
                 symmetric=False, invariant=False):
    """ Construct the off-diagonal elements in a matrix with pair-wise predictions

    When symmetry is set to True, the matrix is symmetrized by M'_ij = (M_ij+M_ji)
    When invariant is set to True, the diagonal elements is filled with the negative
    sum of each column, such that the matrix product of M with R will be translational
    invariant.
    """
    ind_bij = pair_rind
    mat = tf.scatter_nd(ind_bij, params, [nbatch, nmax, nmax])
    if symmetric:
        ind_bji = tf.stack([ind_bij[:,i] for i in [0,2,1]],axis=1)
        mat += tf.scatter_nd(ind_bji, params, [nbatch, nmax, nmax])
    if invariant:
        mat = tf.linalg.set_diag(mat, -tf.reduce_sum(mat, axis=1))
    return mat


def make_dummy(atom_rind, nbatch, nmax):
    """Fill the off-diagonal parts with ones"""
    return tf.eye(nmax, batch_shape=[nbatch])  -  \
           make_diag(atom_rind, tf.ones(tf.shape(atom_rind)[0]), nbatch, nmax)

def make_Dfield_eta(eta, R, cell, atom_rind):
    z = R[:,:,2:]
    
    Omega = tf.math.abs(tf.linalg.det(cell))
    pi = tf.constant(np.pi)
    pi_Omega = tf.divide(tf.multiply(4.0, pi), Omega)[:, tf.newaxis, tf.newaxis]
    
    A = eta
    
    zz = tf.einsum('bia, bja -> bij', z, z)
    A_Dfield = A + tf.multiply(zz, pi_Omega)

    return A_Dfield

def pol_corr_plot(pol_pred, pol_data):
    f, axs = plt.subplots(2, 2,
                          gridspec_kw={
                              'wspace': 0, 'hspace':0,
                              'width_ratios': [3, 1.2],
                              'height_ratios': [1, 3]},
                          figsize=[4,4],
                          sharey='row', sharex='col')
    axs[0,1].axis('off')

    s = 0.5
    axs[1,0].scatter(pol_data[:,[0,1,2],[0,1,2]],
                     pol_pred[:,[0,1,2],[0,1,2]],
                     color='tab:blue', label='diagonal', lw=0, s=s)
    axs[1,0].scatter(pol_data[:,[0,0,1],[1,2,2]],
                     pol_pred[:,[0,0,1],[1,2,2]],
                     color='tab:green', label='off-diag', lw=0, s=s)

    log = False
    lims = [-40,180]
    bins=np.linspace(*(lims+[50]))
    axs[0,0].hist([pol_data[:,[0,1,2],[0,1,2]].flatten(),
                   pol_data[:,[0,0,1],[1,2,2]].flatten()],
                  label=['diagonal', 'off-diag'],
                  color=['tab:blue', 'tab:green'],
                  bins=bins, stacked=True, log=log)
    axs[1,1].hist([pol_pred[:,[0,1,2],[0,1,2]].flatten(),
                   pol_pred[:,[0,0,1],[1,2,2]].flatten()],
                  label=['diagonal', 'off-diag'],
                  color=['tab:blue', 'tab:green'],
                  bins=bins, stacked=True, log=log,
                  orientation='horizontal')
    axs[1,0].set_xlim(*lims)
    axs[1,0].set_ylim(*lims)
    axs[0,0].legend(ncol=1,loc="upper left",
                    bbox_to_anchor=[1, 1])
    axs[0,0].get_yaxis().set_ticks([])
    axs[0,1].get_xaxis().set_ticks([])
    axs[1,0].set_xlabel('$\\alpha$ labels [bohr$^3$]')
    axs[1,0].set_ylabel('$\\alpha$ predictions [bohr$^3$]')

def egap_corr_plot(data,pred,name,xmin=0.1,xmax=0.4):
    f, axs = plt.subplots(2, 2,
                          gridspec_kw={
                              'wspace': 0, 'hspace':0,
                              'width_ratios': [3, 1.2],
                              'height_ratios': [1, 3]},
                          figsize=[4,4],
                          sharey='row', sharex='col')
    axs[0,1].axis('off')

    s = 6
    axs[1,0].scatter(data,pred,color='tab:blue', lw=0, s=s)

    log = False
    bins=52
    axs[0,0].hist(data,color=['tab:blue'],bins=bins, stacked=True, log=log)
    axs[1,1].hist(pred,color=['tab:blue'],bins=bins, stacked=True, log=log,orientation='horizontal')
    axs[1,0].plot(np.arange(xmin,xmax+0.1,step=0.1),np.arange(xmin,xmax+0.1,step=0.1),'k--')
    axs[1,0].set_xlim(xmin,xmax)
    axs[1,0].set_ylim(xmin,xmax)
    axs[0,0].get_yaxis().set_ticks([])
    axs[0,1].get_xaxis().set_ticks([])
    axs[1,0].set_xlabel('HOMO-LUMO labels (a.u.)')
    axs[1,0].set_ylabel('HOMO-LUMO predictions (a.u.)')
    plt.savefig(name,bbox_inches = 'tight')

def planar(tensors, tol=1e-3):
    # whether a molecular is planar (or linear)
    R = tensors['coord']
    R = R - tf.reduce_mean(R, axis=0, keepdims=0)
    cov = tf.transpose(R) @ R
    s, u, v = tf.linalg.svd(cov)
    return s[-1]<tol


def planar2xy(tensors):
    # rotate R->R' such that (R'^TR')[2,2]=0 or R'[:,2]=0
    # let s,u,v = svd(R^TR); if planar, min(s)=0
    # R = R'v; R' = Rv^{-1} = Rv^T = Ru
    R = tensors['coord']
    R = R - tf.reduce_mean(R, axis=0, keepdims=0)
    cov = tf.transpose(R) @ R
    s, u, v = tf.linalg.svd(cov)
    tensors['coord'] = R@u
    if 'ptensor' in tensors:
        tensors['ptensor'] =tf.transpose(u)@tensors['ptensor']@u
    return tensors

def thickness(tensors, tol=1e-3):
    # whether a molecular is planar (or linear)
    R = tensors['coord']
    R = R - tf.reduce_mean(R, axis=0, keepdims=0)
    cov = tf.transpose(R) @ R
    s, u, v = tf.linalg.svd(cov)
    return s[-1]

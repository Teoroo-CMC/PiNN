# -*- coding: utf-8 -*-
r"""
This file implements a combination of the atomic charge dipole model and the bond charge dipole
model with regularization resulting in the AC+BC(R) dipole model.

Atomic property predictions from the network are interpreted as atomic charges.
A charge neutrality constraint is applied to enfore a charge of zero on the 
charge neutral entity by equivalently spreading the excess charge out over the atoms. 
Atomic pairwise interaction predictions from the network are interpreted as bond charges.
The dipole moment is expressed as: 

$$
\begin{aligned}
\mu = \sum_{i}{}^{1}\mathbb{P}_{i} \cdot \mathbf{r}_{i} + \sum_{ij}{}^{1}\mathbb{I}_{ij} \cdot \mathbf{r}_{ij}
\end{aligned}
$$

During the training process, L2-regularization is applied to the atomic pairwise interactions.

This model fits the total dipole of the inputs and predicts the total dipole.
For model details see ref. 
Li, J., Knijff, L., Zhang, Z., Andersson, L., & Zhang, C. (2024)
PiNN: equivariant neural network suite for modelling electrochemical systems
"""
import numpy as np
import tensorflow as tf
from pinn import get_network
from pinn.utils import pi_named
from pinn.models.base import export_model, get_train_op, MetricsCollector

default_params = {
    ### Scaling and units
    # The loss function will be MSE((pred - label) * scale)
    # For vector/tensor predictions
    # the error will be pre-component instead of per-atom
    # d_unit is the unit of dipole to report w.r.t the input labels
    'd_scale': 1.0,  # dipole scale for prediction
    'd_unit': 1.0,  # output unit of dipole during prediction
    # Toggle whether to use scalar or vector dipole predictions
    'vector_dipole': False,
    # Enable charge neutrality
    'charge_neutrality': True,
    # Enable charge neutrality: 
    # 'system' for system wide neutrality
    # 'water_molecule' for neutrality per water molecule
    'neutral_unit': 'system',
    # Enable regularization
    'regularization': True,
    # Loss function options
    'max_dipole': False,     # if set to float, omit dipoles larger than it
    'use_d_per_atom': False,  # use d_per_atom to calculate d_loss
    'log_d_per_atom': False,  # log d_per_atom and its distribution
                             # ^- this is forcely done if use_d_per_atom
    'use_d_weight': False,   # scales the loss according to d_weight
    # L2 loss
    'use_l2': False,
    # Loss function multipliers
    'd_loss_multiplier': 1.0,
    'q_loss_multiplier': 1.0
}

@export_model
def AC_BC_R_dipole_model(features, labels, mode, params):
    r"""The AC+BC(R) constructs the dipole moment from 
    atomic charge predictions and atomic pairwise interactions:

    $$
    \begin{aligned}
    \mu = \sum_{i}{}^{1}\mathbb{P}_{i} \cdot \mathbf{r}_{i} + \sum_{ij}{}^{1}\mathbb{I}_{ij} \cdot \mathbf{r}_{ij}
    \end{aligned}
    $$

    L2-regularization is applied to the atomic pairwise interactions.
    """
    if params['network']['name'] == "PiNet":
        params['network']['params'].update({'out_prop':1, 'out_inter':1})
    
    network = get_network(params['network'])
    model_params = default_params.copy()
    model_params.update(params['model']['params'])

    features = network.preprocess(features)

    if params['network']['name'] == "PiNet2":
        p1, output_dict = network(features)

        i1 = output_dict['i1']
        i3 = output_dict['i3']

        ipred =  tf.einsum("ixr,ixr->ir", i3, i3) + i1

    else:
        p1, ipred = network(features)
    
    ind1 = features['ind_1']  # ind_1 => id of molecule for each atom
    ind2 = features['ind_2']

    natoms = tf.reduce_max(tf.shape(ind1))
    nbatch = tf.reduce_max(ind1)+1 

    if model_params['charge_neutrality'] == True:
        if model_params['neutral_unit'] == 'system':
            q_molecule = tf.math.unsorted_segment_sum(p1, ind1[:, 0], nbatch)
            N = tf.math.unsorted_segment_sum(tf.ones_like(ind1, tf.float32), ind1, tf.reduce_max(ind1)+1)
        
            p_charge = q_molecule/N
            charge_corr = tf.gather(p_charge, ind1)[:,0]
            p1 =  p1 - charge_corr

        if model_params['neutral_unit'] == 'water_molecule':
            q_molecule = tf.math.reduce_sum(tf.reshape(p1,[-1,3]),axis=1)
    
            p_charge = q_molecule/3
            charge_corr = tf.reshape(tf.stack([p_charge, p_charge, p_charge], axis=1), [1,-1])[0,:]
            p1 =  p1 - charge_corr

    p1 = tf.expand_dims(p1, axis=1)
    
    q_tot = tf.math.unsorted_segment_sum(p1, ind1[:, 0], nbatch)
    
    q_d_a = p1 * features['coord']
    q_d = tf.math.unsorted_segment_sum(q_d_a, ind1[:, 0], nbatch)
    
    # Compute bond vector
    disp_r = features['diff']

    # Compute atomic dipole
    atomic_d_pairwise = ipred * disp_r
    atomic_d_a = tf.math.unsorted_segment_sum(atomic_d_pairwise, ind2[:, 0], natoms) 
    atomic_d = tf.math.unsorted_segment_sum(atomic_d_a, ind1[:, 0], nbatch)

    a_dipole = q_d_a + atomic_d_a
    dipole = q_d + atomic_d

    if model_params['vector_dipole'] == False:
        dipole = tf.sqrt(tf.reduce_sum(dipole**2, axis=1)+1e-6)

    if mode == tf.estimator.ModeKeys.TRAIN:
        metrics = make_metrics(features, dipole, q_tot, ipred, model_params, mode)
        tvars = network.trainable_variables
        train_op = get_train_op(params['optimizer'], metrics, tvars)
        return tf.estimator.EstimatorSpec(mode, loss=tf.reduce_sum(metrics.LOSS),
                                          train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = make_metrics(features, dipole, q_tot, ipred, model_params, mode)
        return tf.estimator.EstimatorSpec(mode, loss=tf.reduce_sum(metrics.LOSS),
                                          eval_metric_ops=metrics.METRICS)
    else:
        dipole = dipole / model_params['d_scale']
        dipole *= model_params['d_unit']

        predictions = {
            'dipole': dipole,
            #'charge': q_tot
            #'atomic_d': tf.expand_dims(a_dipole, 0)
        }
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions)


@pi_named("METRICS")
def make_metrics(features, d_pred, q_pred, ipred, params, mode):
    metrics = MetricsCollector(mode)

    d_data = features['d_data']
    d_data *= params['d_scale']
    d_mask = tf.abs(d_data) > params['max_dipole'] if params['max_dipole'] else None
    d_weight = params['d_loss_multiplier']
    d_weight *= features['d_weight'] if params['use_d_weight'] else 1

    metrics.add_error('D', d_data, d_pred, mask=d_mask, weight=d_weight,
                      use_error=(not params['use_d_per_atom']))
  
    if params['regularization'] == True:
        metrics.add_error('ipred', tf.zeros_like(ipred), ipred, log_error=False, log_hist=False, use_error=True)

    q_data = tf.zeros_like(q_pred)
    q_weight = params['q_loss_multiplier']
    metrics.add_error('Total q', q_data, q_pred, weight=0, use_error=False)

    if params['use_d_per_atom'] or params['log_d_per_atom']:
        n_atoms = count_atoms(features['ind_1'], dtype=d_data.dtype)
        metrics.add_error('D_PER_ATOM', d_data/n_atoms, d_pred/n_atoms, mask=d_mask,
                          weight=d_weight, use_error=params['use_d_per_atom'],
                          log_error=params['log_d_per_atom'])

    if params['use_l2']:
        tvars = tf.compat.v1.trainable_variables()
        l2_loss = tf.add_n([
            tf.nn.l2_loss(v) for v in tvars if
            ('bias' not in v.name and 'noact' not in v.name)])
        l2_loss = l2_loss * params['l2_loss_multiplier']
        metrics.METRICS['METRICS/L2_LOSS'] = l2_loss
        metrics.LOSS.append(l2_loss)

    return metrics

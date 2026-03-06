import tensorflow as tf 
from pinn import get_network 
from pinn.utils import pi_named 
from pinn.models.base import export_model, get_train_op, MetricsCollector 
from pinn.utils import count_atoms

r'''
The atomic charge (AC) quadrupole model constructs the quadrupole moment from 
atomic charge predictions.

Qαβ (r) = ∑ qi · (3rαi · rβi − δαβ · |ri|^2),
where q is the atomic charge, r are atomic coordinates and α, β ∈ {x, y, z}.

The definition of traceless quadrupole is Q'ij = 3Qij - tr(Q)δij.

All properties are in atomic units.
'''
    
default_params = {
    ### Scaling and units
    # The loss function will be MSE((pred - label) * scale)
    # For vector/tensor predictions
    # the error will be pre-component instead of per-atom
    # d_unit is the unit of dipole to report w.r.t the input labels
    'quad_scale': 1.0,  # quadrupole scale for prediction
    'q_unit': 1.0,  # output unit of quadrupole during prediction
    # Enable charge neutrality
    'charge_neutrality': True,
    # Set what kind of charge neutrality should be enforced: 
    # 'system' for system wide neutrality
    # 'water_molecule' for neutrality per water molecule
    'neutral_unit': 'system',
    # Loss function options
    'max_dipole': False,     # if set to float, omit dipoles larger than it
    'use_quad_per_atom': False,  # use quad_per_atom to calculate d_loss
    'log_quad_per_atom': False,  # log quad_per_atom and its distribution
                             # ^- this is forcely done if use_quad_per_atom
    'use_quad_weight': False,   # scales the loss according to quad_weight
    # L2 loss
    'use_l2': False,
    # Loss function multipliers
    'quad_loss_multiplier': 1.0, # quadrupole
    'q_loss_multiplier': 1.0 #total charge (not used in loss function)
}

@export_model
def AC_quadrupole_model(features, labels, mode, params):

    network = get_network(params['network'])
    model_params = default_params.copy()
    model_params.update(params['model']['params'])

    features = network.preprocess(features)
    p1 = network(features) #predicted charges [Q]

    ind1 = features['ind_1']  # ind_1 => id of molecule for each atom

    natoms = tf.reduce_max(tf.shape(ind1))
    nbatch = tf.reduce_max(ind1)+1


    if model_params['charge_neutrality'] == True:
      if model_params['neutral_unit'] == 'system':
          q_molecule = tf.math.unsorted_segment_sum(p1, ind1[:, 0], nbatch)
          N = tf.math.unsorted_segment_sum(tf.ones_like(ind1, tf.float32), ind1, tf.reduce_max(ind1)+1) #Yota
      
          p_charge = q_molecule/N
          charge_corr = tf.gather(p_charge, ind1)[:,0]
          p1 =  p1 - charge_corr


    q_tot = tf.math.unsorted_segment_sum(p1, ind1[:, 0], nbatch) #Gets total charge for each molecule, nbatch is for dimensions

    squared_coord = tf.math.reduce_sum(tf.math.square(features['coord']), axis=1) # [r**2]
    q_q = p1 * squared_coord # [q * r**2]
    
    traceless_correction = tf.math.unsorted_segment_sum(q_q, ind1[:,0], nbatch) #∑ qi * ri**2
    
    q_xx = p1 * tf.math.square(features['coord'][:, 0]) #[q * x**2]
    q_yy = p1 * tf.math.square(features['coord'][:, 1]) #[q * y**2]
    q_zz = p1 * tf.math.square(features['coord'][:, 2]) #[q * z**2]

    q_xy = p1 * features['coord'][:, 0] * features['coord'][:, 1] #[q * xy]
    q_xz = p1 * features['coord'][:, 0] * features['coord'][:, 2] #[q * xz]
    q_yz = p1 * features['coord'][:, 1] * features['coord'][:, 2] #[q * yz]

    #Q_aa = 3 * ∑ qi * r_αi**2 - ∑ qi * ri**2, α ∈ {x, y, z}

    Q_xx = 3 * tf.math.unsorted_segment_sum(q_xx, ind1[:,0], nbatch)- traceless_correction
    Q_yy = 3 * tf.math.unsorted_segment_sum(q_yy, ind1[:,0], nbatch) - traceless_correction
    Q_zz = 3 * tf.math.unsorted_segment_sum(q_zz, ind1[:,0], nbatch) - traceless_correction

    #Q_ab = 3 * ∑ qi * r_αi * r_βi, α,β ∈ {x, y, z}
    
    Q_xy = 3 * tf.math.unsorted_segment_sum(q_xy, ind1[:,0], nbatch)
    Q_xz = 3 * tf.math.unsorted_segment_sum(q_xz, ind1[:,0], nbatch)
    Q_yz = 3 * tf.math.unsorted_segment_sum(q_yz, ind1[:,0], nbatch)
    
    quadrupole = tf.stack((Q_xx, Q_yy, Q_zz, Q_xy, Q_xz, Q_yz), axis=1)

    if mode == tf.estimator.ModeKeys.TRAIN:
        metrics = make_metrics(features, quadrupole, q_tot, model_params, mode)
        tvars = network.trainable_variables
        train_op = get_train_op(params['optimizer'], metrics, tvars)
        return tf.estimator.EstimatorSpec(mode, loss=tf.reduce_sum(metrics.LOSS),
                                          train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = make_metrics(features, quadrupole, q_tot, model_params, mode)
        return tf.estimator.EstimatorSpec(mode, loss=tf.reduce_sum(metrics.LOSS),
                                          eval_metric_ops=metrics.METRICS)
    else:
        
        quadrupole = quadrupole / model_params['quad_scale']
        quadrupole *= model_params['d_unit']

        predictions = {
            'quadrupole': quadrupole, 
            'charge': q_tot,
            'charges': tf.expand_dims(p1, 0)
        
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions)


@pi_named("METRICS")
def make_metrics(features, d_pred, q_pred, params, mode):
    metrics = MetricsCollector(mode)

    quadrupole_data = features['quadrupole_data'] 
    quadrupole_data *= params['quad_scale']
    d_mask = tf.abs(quadrupole_data) > params['max_dipole'] if params['max_dipole'] else None
    quad_weight = params['quad_loss_multiplier']
    quad_weight *= features['quad_weight'] if params['use_quad_weight'] else 1

    metrics.add_error('Q', quadrupole_data, d_pred, mask=d_mask, weight=quad_weight,
                      use_error=(not params['use_quad_per_atom']))

    q_data = tf.zeros_like(q_pred)
    q_weight = params['q_loss_multiplier']
    metrics.add_error('Total q', q_data, q_pred, weight=0, use_error=False)

    if params['use_quad_per_atom'] or params['log_quad_per_atom']:
        n_atoms = count_atoms(features['ind_1'], dtype=quadrupole_data.dtype)
        metrics.add_error('QUAD_per_ATOM', quadrupole_data/n_atoms, d_pred/n_atoms, mask=d_mask,
                          weight=quad_weight, use_error=params['use_quad_per_atom'],
                          log_error=params['log_quad_per_atom'])

    if params['use_l2']:
        tvars = tf.compat.v1.trainable_variables()
        l2_loss = tf.add_n([
            tf.nn.l2_loss(v) for v in tvars if
            ('bias' not in v.name and 'noact' not in v.name)])
        l2_loss = l2_loss * params['l2_loss_multiplier']
        metrics.METRICS['METRICS/L2_LOSS'] = l2_loss
        metrics.LOSS.append(l2_loss)

    return metrics

Dipole model
============

The dipole model requires the same dictionary as input as the potential model.
The only difference is the ``model_params`` that can be set. They are listed below
along with their default values.

model_params
------------

.. code:: python

    ### Scaling and units
    # The loss function will be MSE((pred - label) * scale)
    # For vector/tensor predictions
    # the error will be pre-component instead of per-atom
    # d_unit is the unit of dipole to report w.r.t the input labels
    'd_scale': 1.0, # dipole scale for prediction
    'd_unit': 1.0,  # output unit of dipole during prediction
    ### Loss function options
    'max_dipole': False,     # if set to float, omit dipoles larger than it
    'use_d_per_atom': False, # use d_per_atom to calculate d_loss
    'use_d_per_sqrt': False, # 
    'log_d_per_atom': False, # log d_per_atom and its distribution
                             # ^- this is forcely done if use_d_per_atom
    'use_d_weight': False,   # scales the loss according to d_weight
    'use_l2': False,         # L2 regularization
    ### Loss function multipliers
    'd_loss_multiplier': 1.0,
    'l2_loss_multiplier': 1.0,
    ### Optimizer related
    'learning_rate': 3e-4,   # Learning rate
    'use_norm_clip': True,   # see tf.clip_by_global_norm
    'norm_clip': 0.01,       # see tf.clip_by_global_norm
    'use_decay': True,       # Exponential decay
    'decay_step':10000,      # every ? steps
    'decay_rate':0.999,      # scale by ?

Just like the ``potential_model``, ``dipole_model`` automatically saves ``params``
in a ``params.yml`` file during the creation of the ``estimator``. The dipole model
can then be invoked using ``dipole_model('/path/to/model/')``.

Dipole model as a ASE calculator
--------------------------------

A calculator can be created to use the dipole model to predict dipole moment and 
atomic charges in the following manner:

.. code:: python

    from pinn.models import dipole_model
    from pinn.calculator import PiNN_cal
    calc = PiNN_calc(dipole_model('/path/to/model/'), properties=['dipole', 'charges'])
    dipole_moment = calc.get_dipole_moment(atoms)
    charges = calc.get_charges(atoms)
    

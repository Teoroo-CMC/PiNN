Potential model
===============

``pinn.models.potential_model`` requires one dictionary as input
``params``.  The dictionary typically looks like this:

.. code:: python

   {'model_dir': '/path/to/model',
    'network': 'pinet',
    'network_params': {
        'atom_types':[1, 6, 7, 8, 9]
    },
    'model_params': {
     'learning_rate': 1e-3,
     'e_scale': 627.5,
     'e_dress': dress
    }
   }
   
params
------
   
``params`` contains the entire specification of the model, including
the network and training setups.

- ``model_dir`` is the directory of the estimator, where the training
  log and trained parameters are saved.
- ``network`` could be string: the name an implemented network, or a
  defined network function.
- ``network_params`` are the parameters which will be fed into the
  network function
- ``model_params`` are the parameters that control the potential model.
  Below is a list of those parameters and their default values.


.. code:: python
	  
    ### Loss function options
    'max_energy': False,     # if set to float, omit energies larger than it
    'use_e_per_atom': False, # use e_per_atom to calculate e_loss
    'use_e_per_sqrt': False, # 
    'log_e_per_atom': False, # log e_per_atom and its distribution
                             # ^- this is forcely done if use_e_per_atom
    'use_e_weight': False,   # scales the loss according to e_weight    
    'use_force': False,      # include force in Loss function
    'max_force': False,      # if set to float, omit forces larger than it
    'use_f_weights': False,  # scales the loss according to f_weights
    'use_l2': False,         # L2 regularization
    ### Loss function multipliers
    'e_loss_multiplier': 1.0,
    'f_loss_multiplier': 1.0,
    'l2_loss_multiplier': 1.0,
    ### Optimizer related
    'learning_rate': 3e-4,   # Learning rate
    'use_norm_clip': True,   # see tf.clip_by_global_norm
    'norm_clip': 0.01,       # see tf.clip_by_global_norm
    'use_decay': True,       # Exponential decay
    'decay_step':10000,      # every ? steps
    'decay_rate':0.999,      # scale by ?
    
``potential_model`` automatically saves ``params`` in a ``params.yml``
file during the creation of the ``estimator``. The potential model can
then be invoked using ``potential_model('/path/to/model/')``.

Potential model as a ASE calculator
-----------------------------------

A calculator can be created from a model as simple as:

.. code:: python

    from pinn.models import potential_model	  
    from pinn.calculator import PiNN_calc
    calc = PiNN_calc(potential_modle('/path/to/model/'))
    calc.calculate(atoms)

Energy, forces and stress (with PBC) calculations are implemented for
the ASE calculator.


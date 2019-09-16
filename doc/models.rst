=======
 Models
=======

A model in PiNN referes to an ``estimator`` in TensorFlow. In fact,
the ``pinn.models.potential_model`` function is just a shortcut to use
tf.estimtor.Estimator. The ``estimator`` is specified with a
``model_fn`` function, which, given a set of parameters, defines the
loss function, the training algorithm, metrics and predictions of the
neural network.

As in the PiNN code, ``model_fn`` is decoupled from ``network_fn``:
the "network" cares only about making a prediction for each atom,
while "model" defines the rest. The advantage of this approach is that
a ``model_fn`` can be reused for any ``network_fn``, and vice versa.

If you are interested in modifying the ``model_fn``, you might need to
look into the source code of ``pinn.models``. So far, the only models
implmented in PiNN are the potential model and the dipole model. They define various metrics
and loss functions used in training. They also interface with
the ASE calculator, where the potential model predicts the forces and stresses
using the analytical gradients of the potential energy. The dipole model
predicts dipole moment, and can also predict atomic charges.

Potential model
===============

``pinn.models.potential_model`` requires one dictionary as input
``params``.  The dictionary typically like this:

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
- ``network`` could be string: the name a implemented network, or a
  self-defined network function.
- ``network_params`` are parameters which will be feed into the
  network function
- ``model_params`` are the parameters that controls the potential model.
  Below is a list of those used parameter and their default values.


.. code:: python
	  
    ### Loss function options
    'max_energy': False,     # if set to float, omit energies larger than it
    'use_e_per_atom': False, # use e_per_atom to calculate e_loss
    'use_e_per_sqrt': False, # 
    'log_e_per_atom': False, # log e_per_atom and its distribution
                             # ^- this is forcely done if use_e_per_atom
    'use_e_weight': False,   # scales the loss according to e_weigtht    
    'use_force': False,      # include force in Loss function
    'max_force': False,      # if set to float, omit forces larger than it
    'use_f_weights': False,  # scales the loss according to f_weigthts
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

Dipole model
============

The dipole model requires the same dictionary as input as the potential model.
The only difference is the ``model_params`` that can be set. They are listed below
along with their default values.

model_params
------------

..code:: python

    ### Loss function options
    

Just like the ``potential_model``, ``dipole_model`` automatically saves ``params``
in a ``params.yml`` file during the creation of the ``estimator``. The dipole model
can then be invoked using ``dipole_model('/path/to/model/')``.

Dipole model as a ASE calculator
--------------------------------

A calculator can be created to use the dipole model to predict dipole moment and 
atomic charges in the following manner:

..code:: python

    from pinn.models import dipole_model
    from pinn.calculator import PiNN_cal
    calc = PiNN_calc(dipole_model('/path/to/model/'), properties=['dipole', 'charges'])
    dipole_moment = calc.get_dipole_moment(atoms)
    charges = calc.get_charges(atoms)

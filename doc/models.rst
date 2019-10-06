=======
 Models
=======

Write a model
=============

A model in PiNN refers to an ``estimator`` in TensorFlow. In fact, the
``pinn.models.potential_model`` function is just a shortcut to use
tf.estimtor.Estimator.  The ``estimator`` is specified with a
``model_fn`` function. The ``model_fn`` takes the tensors from the
dataset as input, and should return the training, evaluation or
prediction ``EstimatorSpec`` according to the ``mode`` option.

As in the PiNN code, ``model_fn`` is decoupled from ``network_fn``:
the "network" cares only about making a prediction for each atom,
while the "model" defines the rest. The advantage of this approach is
that a ``model_fn`` can be reused for any ``network_fn``, and vice
versa.

If you are interested in modifying the ``model_fn``, you might need to
look into the source code of ``pinn.models``. So far, the only models
implemented in PiNN are the potential model and the dipole model. They
define various metrics and loss functions used in training. They also
interface with the ASE calculator, where the potential model predicts
the forces and stresses using the analytical gradients of the
potential energy. The dipole model predicts dipole moment, and can
also predict atomic charges.

Implemented models
==================

.. toctree::
   :maxdepth: 1

   models/potential.rst
   models/dipole.rst

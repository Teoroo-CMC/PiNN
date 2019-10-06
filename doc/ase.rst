=============
ASE interface
=============

PiNN provides a ``PiNN_calc`` class to interface with ASE. 

A calculator can be created from a model as simple as:

.. code:: python

    from pinn.models import potential_model	  
    from pinn.calculator import PiNN_calc
    calc = PiNN_calc(potential_modle('/path/to/model/'))
    calc.calculate(atoms)
    
The implemented properties of the calculator depend on the prediciton
returns of ``model_fn``. For example: energy, forces and stress (with
PBC) calculations are implemented for the potential model; partial
charge and dipole calculations are implemented for the dipole model.

The calculator can then be used in ASE optimizers and molecular
dynamics engines. Note that the calculator will try to use the same
predictor (a generator given by ``estimator.predict``) whenever
possible, so as to avoid the expensive reconstruction of the
computation graph. However, the graph will be reconstructed if the pbc
condition of the input ``Atoms`` is changed.  Also, the predictor must
be reset if it is interupted for some reasons.


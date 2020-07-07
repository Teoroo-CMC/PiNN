# Models in PiNN

## Saving and restoring a model

The parameters model is defined by serializable dictionary and saved to a
`params.yml` file whenever a model is created. The `params.yml` file contains
information about the network architecture, hyperparameters and training
algorithm. The `params.yml` file must contain following keys:

- `network`: string specifying a PiNN network or a Keras model
- `network_params`: parameters to initalize the network
- `model_dir`: model directory
- `model_params`: model specific parameters

To restore a model, one only need to initialize the model with the directory:

```Python
from pinn.models import potential_model
model = potential_model('/tmp/pinet_potential')
```

## Common parameters of model

The available options of `model_params` depends on the model, below is a table
of parameters that are present in all models.

| Parameter           | Default           | Description                                              |
|---------------------|-------------------|----------------------------------------------------------|
| `optimizer`         | `AdamOptimizer`   | String specifying optimizer  or a Keras optimizer object |
| `use_l2`            | `False`           | Use the L2 regularization                                |
| `l2_loss_muliplier` | 1.0               | Weight of the L2 loss function in the loss function      |

## ASE interface
PiNN provides a ``PiNN_calc`` class to interface models with ASE. A calculator
can be created from a model as simple as:

```Python
from pinn.models import potential_model
from pinn.calculator import PiNN_calc
calc = PiNN_calc(potential_modle('/path/to/model/'))
calc.calculate(atoms)
```

The implemented properties of the calculator depend on the prediciton returns of
``model_fn``. For example: energy, forces and stress (with PBC) calculations are
implemented for the potential model; partial charge and dipole calculations are
implemented for the dipole model.

The calculator can then be used in ASE optimizers and molecular dynamics
engines. Note that the calculator will try to use the same predictor (a
generator given by ``estimator.predict``) whenever possible, so as to avoid the
expensive reconstruction of the computation graph. However, the graph will be
reconstructed if the pbc condition of the input ``Atoms`` is changed. Also, the
predictor must be reset if it is interupted for some reasons.

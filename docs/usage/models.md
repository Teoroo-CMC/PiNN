# Models in PiNN

For regression problems for an atomic for molecular property, it's often
sufficient to use an `network`. However, atomic machine learning tasks it's
often desired to train on properties derived from the atomic predictions, such
as forces, stress tensors for dipole moments. `pinn.models` are created for
defining these tasks in a `network`-agnostic way. 

Two models are implemented in PiNN at this point, their respective options can
be found in the "Implemented models" section.

## Configuring a model

Models implemented in PiNN used a serialized format for their parameters. The
parameter file specifies the network architecture, hyperparameters and training
algorithm. A typical parameter file include the following sections:

```yaml
model_dir: /tmp/test_pinet_pot
model:
  name: potential_model
  params:
    use_force: true
network:
  name: PiNet
  params:
    atom_types: [1, 6, 7, 8, 9]
optimizer:
  class_name: EKF
  config:
    learning_rate: 0.03
```

Among those, and `optimizer` follows the format of a Keras optimizer. The
`model` and `network` section constitutes the name and parameters of initialize
a PiNN model and network, respectively. A model can be initialized by a
parameter file or a nested python dictionary.

```Python
model = pinn.get_model('pinet.yml')
```

PiNN automatically saves a `params.yml` file in the model directory. With an
trained model, the model can be loaded with its directory as well.

```Python
model = pinn.get_model('/tmp/test_pinet_pot')
```

## ASE interface
PiNN provides a ``PiNN_calc`` class to interface models with ASE. A calculator
can be created from a model as simple as:

```Python
calc = pinn.get_calc_from_model('/tmp/test_pinet_pot')
calc.calculate(atoms)
```

The implemented properties of the calculator depend on the model. For example:
the potential model implements energy, forces and stress (with PBC) calculations
and the dipole model implements partial charge and dipole calculations.

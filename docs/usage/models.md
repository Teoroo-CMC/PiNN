# Models in PiNN

For regression problems for an atomic or molecular property, it's often
sufficient to use an `network`. However, atomic machine learning tasks it's
often desired to train on properties derived from the atomic predictions, such
as forces, stress tensors for dipole moments. `pinn.models` are created for
defining these tasks in a `network`-agnostic way.

Two models are implemented in PiNN at this point, their respective options can
be found in the "Implemented models" section.

## Configuration

Models implemented in PiNN used a serialized format for their parameters. The
parameter file specifies the network architecture, hyperparameters and training
algorithm. A typical parameter file include the following sections:

```yaml
model_dir: pinet_potential
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

Among those, the `optimizer` section follows the format of a Keras optimizer.
The `model` and `network` sections specify the name and parameters of initialize
a PiNN model and network, respectively. A model can be initialized by a
parameter file or a corresponding nested python dictionary.

## Training

The model maybe created by calling the corresponding model function, and a
parameter dictionary mirroring the parameter file:

```Python
import yaml
from pinn.models.potential import potential_model
with open('params.yml') as f:
    params = yaml.load(f, Loader=yaml.Loader)
model = potential_model(params)
```

PiNN provides a shortcut `pinn.get_model` to create an implemented model from a
parameter dictionary or parameter file.

```Python
model = pinn.get_model('params.yml')
```

`pinn.get_model` automatically saves a copy `params.yml` file in the model
directory. When such a file exist, the model can be loaded with its directory as
well.

```Python
model = pinn.get_model('pinet_potential')
```

The PiNN model is a TensorFlow estimator, to train the model in a python script:

```Python
filelist = glob('{DATASET_PATH}/QM9/dsgdb9nsd/*.xyz')
dataset = lambda: load_qm9(filelist, splits={'train':8, 'test':2})
train = lambda: dataset()['train'].repeat().shuffle(1000).apply(sparse_batch(100))
test = lambda: dataset()['test'].repeat().apply(sparse_batch(100))
train_spec = tf.estimator.TrainSpec(input_fn=train, max_steps=1000)
eval_spec = tf.estimator.EvalSpec(input_fn=test, steps=100)
tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
```

## ASE interface
PiNN provides a ``PiNN_calc`` class to interface models with ASE. A calculator
can be created from a model as simple as:

```Python
calc = pinn.get_calc('pinet_potential')
calc.calculate(atoms)
```

The implemented properties of the calculator depend on the model. For example:
the potential model implements energy, forces and stress (with PBC) calculations
and the dipole model implements partial charge and dipole calculations.


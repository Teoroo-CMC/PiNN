# Quick start
## Installation
**using pip**

```bash
git clone https://github.com/Teoroo-CMC/PiNN.git -b TF2
pip install -e PiNN[gpu]
pinn -h
```

Extra dependencies can be specified:

- `[gpu]` or `[cpu]`: cpu or gpu version of TensorFlow
- `[dev]`: development requirements for testing
- `[doc]`: documentation requirements
- `[extra]`: extra requirements (Jupyter, pymatgen, etc) 


**using container** 

PiNN provides two built docker images, which can be converted to singularity
images without much effort:

```bash
singularity build pinn.sif docker://teoroo/pinn:master-gpu
./pinn.sif -h
```

- `:latest-cpu` is much smaller, it comes without GPU support
- `:latest-gpu` is the version with GPU support

Extra dependencies like `Jupyter` are included in the image, for a quick 
development environment:

```bash
singularity run pinn.sif jupyter notebook
```

## Configuration
In PiNN, a model consists of two essential parts, the network and the model. The
network specifies the neural network architecture used to product atomistic
predictions from the input (coordinates and elements information). While the
model interprets the predictions as physical quantities.

In PiNN, the parameters for a model is saved as a `yaml` formatted parameter
file. A minimal example of model parameter file looks like this:

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
```

In addition to the network and the model, the training algorithm should be specified
as such:
```yaml
optimizer:
  class_name: EKF
  config:
    learning_rate: 0.03
```

## Using the CLI
PiNN provides a CLI for training and simple dataset operations:

```bash
pinn convert data.traj -o 'trian:8,eval:2'
pinn train -d model -t train.yml -e eval.yml -b 30 --train-steps 100000 params.yml
```

The above command takes a trajectory file `data.traj`, and splits it into two
datasets. Then it takes the parameters from `params.yml`, and trains a model
in the `model` directory.

## Monitoring
PiNN uses TensorFlow as a backend for training, which means the training log can 
be monitored in real time using Tensorboard:
```bash
tensorboard --log-dir model --port 6006
```

Or with the log inspector of PiNN:
```bash
pinn log model
```


## Using the model
The simplest use case for PiNN models is to use them as ASE calculators. A calculator
can be initialized from the model directory.

```python
import pinn
calc = pinn.get_calc('model')
calc.calculate(atoms)
```

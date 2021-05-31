# Quick start
## Installation
**using pip**

```bash
git clone https://github.com/Teoroo-CMC/PiNN.git -b TF2
pip install -e PiNN[gpu]
pinn -h
```

Extra dependencies are avaiable:

- `[gpu]` and `[cpu]`: cpu and gpu version of Tensorflow
- `[dev]`: development requirements for testing
- `[doc]`: documentation requirements
- `[extra]`: extra requirements (Jupyter, pymatgen, etc) 


**using container** 

PiNN provides three built docker images, which can be converted to 
singularity images without much effort:

```bash
singularity build pinn.sif docker://teoroo/pinn:latest-gpu
./pinn.sif -h
```

Two images are provided:

- `:latest-cpu` is much smaller, it comes without GPU support
- `:latest-gpu` is the version with GPU support

Extra dependencies like `Jupyter` is included in the image, for a quick 
development envrionment:
```bash
singularity run pinn.sif jupyter notebook
```

## Configuration
In PiNN, a model consists two essential parts, the network and the model. The
network specifies the neural network architecture used to product atomistic
predictions from the input (coordinates and elements information). While the
model interprets the predictions as physical quantities.

In PiNN, the parameters for a model is saved as a `yaml` formated parameter
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
pinn train -t train.yml -e eval.yml -b 30 --train-steps 100000 params.yml
```

## Monitoring
PiNN uses TensorFlow as a backend for training, which means the training log can 
be monitored in real time using Tensorboard:
```bash
tensorboard --model-dir model --port 6006
```

Or with the log inspector of PiNN:
```bash
pinn log model
```


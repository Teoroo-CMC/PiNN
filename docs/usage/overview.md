# Using PiNN

Below is a brief introduction of what is implemented in PiNN. In short, the two
core components of PiNN are the **networks** - machine learning models that
generates prediction from atomic structures; and **models** which interprets
those predictions for different tasks.

## Networks

Atomic neural **networks** (ANN) in PiNN are written in Keras. The "networks"
are Keras `Models` which takes atomic structures as input and outputs atomic
predictions. `Layers` are components of `Models` or networks.

`pinn.layers` contains reusable operations in bulding ANNs, e.g. calculation of
neighboring lists, or radial basis functions. `pinn.networks` contains ANN
implementations. Since PiNN `networks` are essentially Keras Models, they are
ready for simple regression tasks, see an example
[here](https://colab.research.google.com/github/yqshao/PiNNLab/blob/master/notebooks/7_pinn.ipynb).

## Models

In addition to layers and networks, PiNN implement several **models**. PiNN
models interpret the output of ANNs as physical quantities, e.g. atomic
energies. Those `models` enables the training of quantities derives from the
atomic predictions, like forces (from energy) and dipole moments (from partial
charges).

PiNN `models` are implemented as TensorFlow estimators. `models` are also
responsible for interfacing with external libraries like `ASE` to run
simulations. When training `models` it is recommended to use the CLI interface
of PiNN.

## What to read

Checkout [quick start](quick_start.md) to get started. See the
[notebook](../notebooks/overview.md) examples for more examples.

If you are interested in a specific application, e.g. fitting a machine-learned
potential, read the available options of the [potential model](potential.md) and
network (likely [PiNet](pinet.md)) you'd like to use.

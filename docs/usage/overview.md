# Using PiNN

## Architecture

Components of PiNN are written in the language of Keras Layers and Models, those
are referred to as PiNN `layers` and `networks` respectively.

PiNN `layers` are reusable operations in training ANNs, e.g. calculation of
neighboring lists, or radial basis functions. 

PiNN `networks` are defined ANN architectures that makes atomic predictions.
Since PiNN `networks` are essentially Keras Models, they are ready for simple
regression tasks.

## Models

In addition to layers and networks, PiNN implement several models. PiNN `models`
interpret the output of ANNs as physical quantities, e.g. atomic energies. Those
`models` enables the training of quantities derives from the atomic predictions,
like forces and dipole moments. 

PiNN `models` are implemented as TensorFlow estimators. `models` are also
responsible for interfacing with external libraries like `ASE` to run
simulations.

## What to read

Checkout [quick start](quick_start.md) to get started. See the
[notebook](../notebooks/overview.md) examples for more examples.

If you are interested in a specific application, e.g. fitting a machine-learned
potential, read the available options of the [potential model](potential.md) and
network (likely [PiNet](pinet.md)) you'd like to use.

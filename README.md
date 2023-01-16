# PiNN: a Python library for building atomic neural networks

![GitHub Workflow Status (branch)](https://img.shields.io/github/actions/workflow/status/Teoroo-CMC/PiNN/build_and_test.yml?branch=master&label=build&style=flat-square)

PiNN<sup>[1](#fn1)</sup> is a Python library built on top of TensorFlow for
building atomic neural network potentials. The PiNN library also provides
elemental layers and abstractions to implement various atomic neural networks.

The code is currently maintained by Yunqi Shao at Uppsala University.

## Requirements

- Python >= 3.7
- [ASE](https://wiki.fysik.dtu.dk/ase/) >= 3.19
- [NumPy](https://numpy.org/) >= 1.15
- [PyYAML](https://pyyaml.org/) > 3.01
- [TensorFlow](https://www.tensorflow.org/install) >= 2.4<sup>[2](#fn2)</sup>

## Installation

Install from source code::

``` sh
git clone https://github.com/Teoroo-CMC/PiNN.git 
cd PiNN && pip install -e .
```

Or use the [docker
image](https://cloud.docker.com/repository/docker/teoroo/pinn/tags). If you use
singularity, you can build a singularity image directly from the docker image:

``` sh
singularity build pinn.sif docker://teoroo/pinn:master-gpu (or master-cpu)
singularity exec pinn.sif jupyter notebook # this starts a jupyter notebook server
./pinn.sif --help # this invokes the pinn CLI
```

## Documentation

Since PiNN 1.0 the documentation is hosted on [Github pages](https://teoroo-cmc.github.io/PiNN/)

## Models and datasets

### Dataset loaders

- CP2K format
- RuNNer format
- ANI-1 dataset
- QM9 dataset

### Implemented Networks

- PiNet
- Behler-Parrinello Neural Network

### Implemented models

- Potential model
- Dipole model

## Community

As an open-source project, the following contributions are highly welcome:

- Reporting bugs
- Proposing new features
- Discussing the current version of the code
- Submitting fixes

We use Github to host code, to track issues and feature requests, as well
as to accept pull requests. 

Please follow the procedure below before you open a new issue.

- Check for duplicate issues first.
- If you are reporting a bug, include the system information
  (platform, Python and TensorFlow version etc.).

If you would like to add some new features via pull request, please
discuss with the main developer (Yunqi Shao) first to see whether it
fits the scope and aims of this project.

## References and notes

<a name="fn1">[1]</a> Shao, Y.; Hellström, M.; Mitev, P. D.; Knijff, L.; Zhang,
C. PiNN: A Python Library for Building Atomic Neural Networks of Molecules and
Materials. arXiv:1910.03376 [cond-mat, physics:physics] 2019.

<a name="fn2">[2]</a> TensorFlow is not installed automatically by default.
Since TF 2.0 the GPU support is included in the stable release, ``pip install
tensorflow>=2.4`` should be suitable for most user.


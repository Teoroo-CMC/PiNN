# PiNN: Pair-wise interaction Neural Network

![GitHub Workflow Status (branch)](https://img.shields.io/github/actions/workflow/status/Teoroo-CMC/PiNN/build_and_test.yml?branch=master&label=build&style=flat-square)

PiNN<sup>[1](#fn1),[2](#fn2)</sup> is a pair-wise interaction neural network Python library built on top of TensorFlow. The PiNN library provides elemental layers and abstractions to implement various atomic neural networks. It can be used together with plugins [PiNNAcLe](https://github.com/Teoroo-CMC/PiNNAcLe) for the adaptive learn-on-the-fly workflow and [PiNNwall](https://github.com/Teoroo-CMC/PiNNwall) for molecular simulation of electrode/electrolyte interfaces.

This project was initiated by [Yunqi Shao][yqshao]. The code is currently maintained by the [TeC group][tec] at Uppsala University.

[yqshao]:https://github.com/yqshao
[tec]:https://tec-group.github.io/

## Requirements

- Python >= 3.7
- [ASE](https://wiki.fysik.dtu.dk/ase/) >= 3.19
- [PyYAML](https://pyyaml.org/) > 3.01
- [TensorFlow](https://www.tensorflow.org/install) >= 2.4<sup>[3](#fn3)</sup> and <=2.9<sup>[4](#fn4)</sup>

## Installation

We recommend two ways to install PiNN.

1) You need to first create a virtual environment:
``` sh
git clone https://github.com/Teoroo-CMC/PiNN.git
cd PiNN
```

``` sh
conda env create -f environment.yml
```
After activating the pinn environment, then install PiNN using the following command:

``` sh
pip install -e .
```

2) Alternatively, you can use the [docker
image](https://hub.docker.com/r/tecatuu/pinn/tags) to contain the environment. If you prefer
singularity, you need build a singularity image from the docker image:

``` sh
singularity build pinn.sif docker://tecatuu/pinn:master-gpu (or master-cpu)
singularity exec pinn.sif jupyter notebook # this starts a jupyter notebook server
./pinn.sif --help # this invokes the pinn CLI
```

## Documentation

Since PiNN 1.0 the documentation is hosted on [Github pages](https://teoroo-cmc.github.io/PiNN/)

## Models and datasets

### Dataset loaders

- CP2K format
- RuNNer format
- ANI-1 format
- QM9 format
- DeePMD-kit format

### Implemented Networks

- Behler-Parrinello Neural Network
- PiNet
- PiNet2

### Implemented models

- Potential model
- Dipole model
- Polarizability model

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
discuss with us first to see whether it fits the scope and aims of this project.

## References and notes

<a name="fn2">[1]</a> Li, J.; Knijff, L.; Zhang, Z.-Y.; Andersson, L.; Zhang,
C. PiNN: Equivariant Neural Network Suite for Modelling Electrochemical Systems. J. Chem. Theory Comput., 2025, 21: 1382.

<a name="fn1">[2]</a> Shao, Y.; Hellström, M.; Mitev, P. D.; Knijff, L.; Zhang,
C. PiNN: A Python Library for Building Atomic Neural Networks of Molecules and
Materials. J. Chem. Inf. Model., 2020, 60: 1184. 

<a name="fn3">[3]</a> TensorFlow is not installed automatically by default.
Since TF 2.0 the GPU support is included in the stable release, ``pip install
tensorflow>=2.4`` should be suitable for most user. 

<a name="fn4">[4]</a> Currently the code is not compatible with TF 2.10 and above,
see [Issue #7](https://github.com/Teoroo-CMC/PiNN/issues/7) for details or updates.

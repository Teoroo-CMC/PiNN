=========================================
PiNN: Pairwise interaction Neural Network
=========================================

.. image:: https://img.shields.io/circleci/token/14f38a1cab4c2bef74b12be05854d3d62f9c04c3/project/github/Teoroo-CMC/PiNN_dev/dev.svg?style=flat-square
    :target: https://circleci.com/gh/Teoroo-CMC/PiNN_dev/tree/dev
	     
.. image:: https://img.shields.io/codecov/c/github/Teoroo-CMC/PiNN_dev/dev.svg?token=3ab2d943114443d99e92266516befc02&style=flat-square
  :target: https://codecov.io/gh/Teoroo-CMC/PiNN_dev/branch/dev

.. image:: https://img.shields.io/docker/cloud/build/yqshao/pinn.svg?style=flat-square
  :target: https://cloud.docker.com/repository/docker/yqshao/pinn
	   
PiNN is a neural network designed for modeling atomic potentials.
The PiNN package also provides elemental layers and abstractions to implement
various atomic neural networks.

The code is currenly developed by Yunqi Shao at Uppsala Unversiy.

Introduction
============
PiNN is a atomic neural network potential which:

- Requires minimal feature-design
- Is based on solely pairwise interactions
- Preduces state-of-art accurary and speed
  
For more information, see our preprint.(**doesn't exist yet**)

PiNN is also a tool to:

- Train and evaluate different atomic neural networks
- Construct new neural networks with existing building blocks
- Develope new building blocks for ANNs
  
Read the documentation for more details.

Requirements
============
- Python 3
- Tensorflow, ASE
- h5py (optional, for reading ANI-1 dataset)

Installation
============

Install from source code::

  git clone git@github.com:Teoroo-CMC/PiNN_dev.git
  cd PiNN_dev && pip install -e .

Or use our `docker
image <https://cloud.docker.com/repository/docker/yqshao/pinn/tags>`_. If
you use singularity, you can build a singularity image directly from
the docker image::

  singularity build pinn.sif docker://yqshao/pinn:dev (or dev-gpu)
  singularity exec pinn.sif jupyter notebook # this starts a jupyter notebook server
  ./pinn.sif -h # this invokes the pinn_train trainner

Extra dependencies are in:

- ``requirements-dev.txt``: dependency for testing and documentation building.
- ``requirements-extra.txt``: extra libraries for various purposes, included in the docker image.
  
Quick Start
===========
A set of tutorial notebooks can be found in the documentation.

Models and datasets
===================

Implemented Models
------------------
- Graph convolutional based
  
  - PiNet

- Symmtry function based
  
  - Behler-Parrinello Neural Network

Dataset Connectors
------------------

- RuNNer  
- ANI-1
- QM9

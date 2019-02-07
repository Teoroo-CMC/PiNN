=========================================
PiNN: Pairwise interaction Neural Network
=========================================
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
PiNN is currently under development and available via git repo::

  git clone git@github.com:Teoroo-CMC/PiNN_dev.git
  cd PiNN_dev && pip install -e .

Quick Start
===========
A set of tutorial notebooks can be found in the documentation.

Models and datasets
===================

Implemented Models
------------------
- Graph convolutional based
  
  - PiNN
  - (TODO) SchNet/DTNN
  - (TODO) HIPNN 
- Symmtry function based
  
  - (TODO) Behler-Parrinello Neural Network

Dataset Connectors
------------------
- Organic Moleclues
  
  - ANI-1
  - QM9
- Bulk Materials
  
  - (TODO) MP 

References
==========

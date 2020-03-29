==========================================================
PiNN: a Python library for building atomic neural networks
==========================================================

.. image:: https://img.shields.io/circleci/build/github/Teoroo-CMC/PiNN/master.svg?style=flat-square&token=dd8894015481e2f87675a340fbfa712c94d69e8f
   :target: https://circleci.com/gh/Teoroo-CMC/PiNN/tree/master
	     
.. image:: https://img.shields.io/codecov/c/github/Teoroo-CMC/PiNN/master.svg?style=flat-square
   :target: https://codecov.io/gh/Teoroo-CMC/PiNN/branch/master

.. image:: https://img.shields.io/docker/cloud/build/yqshao/pinn.svg?style=flat-square
   :target: https://cloud.docker.com/repository/docker/yqshao/pinn

.. image:: https://readthedocs.org/projects/teoroo-pinn/badge/?version=latest&style=flat-square
   :target: https://teoroo-pinn.readthedocs.io/en/latest/?badge=latest
      
PiNN is a Python library built on top of TensorFlow for building
atomic neural network potentials. The PiNN library also provides
elemental layers and abstractions to implement various atomic neural
networks.

The code is currenly maintained by Yunqi Shao at Uppsala Unversiy.

Reference
=========
- Shao, Y.; Hellström, M.; Mitev, P. D.; Knijff, L.; Zhang, C. PiNN: A
  Python Library for Building Atomic Neural Networks of Molecules and
  Materials. arXiv:1910.03376 [cond-mat, physics:physics] 2019. `link
  <http://arxiv.org/abs/1910.03376>`_

Requirements
============
- Python 3
- ASE, Numpy, Pyyaml
- TensorFlow >= 2.1 [#tf_version]_

Installation
============

Install from source code::

  git clone https://github.com/Teoroo-CMC/PiNN.git
  cd PiNN && pip install -e .

Or use our `docker
image <https://cloud.docker.com/repository/docker/yqshao/pinn/tags>`_. If
you use singularity, you can build a singularity image directly from
the docker image::

  singularity build pinn.sif docker://yqshao/pinn:latest (or latest-gpu)
  singularity exec pinn.sif jupyter notebook # this starts a jupyter notebook server
  ./pinn.sif -h # this invokes the pinn_train trainner

Extra dependencies are in:

- ``requirements-dev.txt``: dependency for testing and documentation building.
- ``requirements-extra.txt``: extra libraries for various purposes, included in the docker image.
  
Quick Start
===========
A set of tutorial notebooks can be found in the `documentation <https://teoroo-pinn.readthedocs.io/en/latest>`_.

Models and datasets
===================

Dataset loaders
---------------
- CP2K format
- RuNNer format
- ANI-1 dataset
- QM9 dataset

Implemented Networks
--------------------
- PiNet
- Behler-Parrinello Neural Network  

Implemented models
------------------
- Potential model
- Dipole model  

Community
=========
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

Notes
=====

.. [#tf_version] TensorFlow is not installed automatically by default. Since TF
                 2.0 the GPU support is included in the stable release, ``pip
                 install tensorflow>=2.1`` should be suitable for most user.
                 This dependency can be included by appending the ``[gpu]``
                 option when installing PiNN with pip. Otherwise, you can
                 install PiNN with CPU-only tensorflow using the ``[cpu]``
                 option.

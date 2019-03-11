Introduction
============

PiNN is a python library built on top of Tensorflow for building
atomic neural networks (ANN).
The primary usage of PiNN is to build and train ANN potentials,
but also capable of predicting other properties from atomci structures.
PiNN is also named after our own :doc:`network structure <networks/pinn>`.


ANN is a machine learning approach to make predictions
from atomic structures.
Early attempt in this direction often requies the while atomic structure
(e.g. coulomb matrix, internal coordinates as input).
This approach were found limited for make general predictions, as the
input size must be fixed.

.. image:: images/global_net.svg

Later approaches switch to more "local" features as inputs
(e.g. atom densities, symmetry functions).
Those networks achieved nice performance and were proven
useful for performing large-scale simulations.


.. image:: images/local_net.svg

After that, more implementations emerged
with varing structures and applications.
Nevertheless, dispite the difference between those models,
they share quite rather similar components
(:ref:`here <layer_types>` is a more detailed discussion):

- interaction between pairs (pi layers)
- obtaining atomic property from interactions (ip layers)
- feed-forward neural networks to fit complex functions (fc layers)
- and needless to say, those models requires similar input/output.

.. image:: images/layers.svg
  
PiNN seeks to provide a framework and optimized components to
build and train those atomic neural networks.
The goal is to be flexible, fast and reliable.

Flexibiliy
^^^^^^^^^^

PiNN is als built with modularized components and we try to make it as easy
as possible to tweak.
So there's no need to rewrite everything if you just want to change
some of the layers, or use existing network for predicting new properties.

Speed
^^^^^

PiNN fully adheres to tensorflow's high-level Estimator and Dataset API.
It is straitforward to train and predict on different compute resources
(CPU, multi-GPU, google cloud, etc) without worring about optimization.

Example
^^^^^^^

The quickest way to start with PiNN is to follow our tutorial :doc:`notebooks`.
The notebooks shall guide you from training a simple ANN potential with
to customize PiNN for your own systems.

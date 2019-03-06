SchNet
======

SchNet :cite:`schutt_schnetdeeplearning_2018` is a graph-convolution
neural network for learning various molecular properties.


Structure
---------

Schent features the continuous-filter convolutional (cfconv) layer.
The cfconv layer can be understood as a distance-dependent filter
which "collectes" information from neighbouring atoms.
The filter is distance-dependent to enforce the rotational invariance.
Cfconv is constructed with a set of radial basis functions
and learnable weights.

Each interation block in SchNet consists of a trainable filter generater
and a atom-wise activation layer, as shown in the illustration.
SchNet consists from `T` interaction blocks, starting with a trainable
embedding layer and ending with a fully-connected atomic layer.

The SchNet implemented in PiNN package is constructed according to the
original paper, :cite:`schutt_schnetdeeplearning_2018`
with a few adjustable parameters to play with.

API Reference
-------------
.. autofunction:: pinn.networks.schnet_network


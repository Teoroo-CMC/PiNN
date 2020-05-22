Customized  Network
===================

One of PiNN's goal is to make it easy to implement different atomic neural
network architectures.

Basic concepts
--------------

Atomic property and pairwise  interactions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Building blocks
^^^^^^^^^^^^^^^

Output
------

The output

Preprocessing
-------------

Preprocess is a reserved method of PiNN networks. This layer should contain the
parameter-free layers of the network architecture, such as the calculation of
neighboring lists and basis funtions. As discussed in overview, the preprocess
method can improve the training speed.

One of the drawback of saving a preprocessed dataset is that the gradient
information is lost. To enable training on the derivatives of the prediction
(namely the forces and stress) in a potential model, the derivatives information
must be supplemented when loading a pre-processed dataset.

In PiNN networks, the preprocess layer works differently when dealing with
preprocessed and raw data. When the data is preprocessed, the preprocess layer
will reconnect the gradients from the coordinates ("coord") to the bond vectors
("diff") to the pairwise distances ("dist").

In more complete cases when atomic fingerprints are computed from the distances
or even angles, gradient of the fingerprints must also be supplies. Implementing
such a preprocess layer is not trivial but possible. Please check out the source
code of pinn.networks.bpnn if you'd like to do so.

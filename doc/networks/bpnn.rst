BPNN
====

A more detailed tutorial about BPNN can be can be found here :cite:`behler_constructinghighdimensionalneural_2015`.

Structure
---------

BPNN features a element-specific network structure,
meaning the description and network parameters are separated.

The atomic environment descriptor or fingerprint is the key component of BPNN.
The descriptors are symmetry functions (SFs),
which are effectively the radial/anglular densities of the environment,
focusing on certain distance/angles.
The set of fingerprints can be specified as such

.. code-block:: python
	       
   [{'G2':[1.0, 1.5, 2.0]}, {'G4':[0.5, 1.0, 2.0]}]

The descriptor can futher depend on the neigbouring atoms

.. code-block:: python
	       
   {a:{b: spec_1, c: spec_2}, b:{...}, ...}

In this case the `spec_1` SFs of b and `spec_2` SFs of c will be concatenated
as the descriptor of element a's environment

API reference
-------------
.. autofunction:: pinn.networks.bpnn_network

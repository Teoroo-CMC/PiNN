# The PiNet network

The PiNet network implements the network architecture described in our
paper.[@2020_ShaoHellstroemEtAl] The network architecture features the
graph-convolution which recursively generates atomic properties from local
environment.

One distinctive feature of PiNet is that the convolution operation is realized
with pairwise functions whose form are determined by the pair, called pairwise
interactions.

## Parameters

| Parameter   | Default         | Description                                                           |
|-------------|-----------------|-----------------------------------------------------------------------|
| atom_types  | `[1, 6, 7, 8] ` | List of elements                                                      |
| rc          | `4.0`           | Cutoff radius                                                         |
| cutoff_type | `'f1'`          | One of 'f1' or 'f2'                                                   |
| basis_type  | `'polynomial'`  | One of 'polynomial' or 'gaussian'                                     |
| n_basis     | `4`             | Number of radial basis functions to generate                          |
| gamma       | `3.0`           | controls width of the Gaussian basis                                  |
| center      | `None`          | replace centers the for Gaussian basis with a list                    |
| pp_nodes    | `[16, 16]`      | specifies the property-property hidden layers                         |
| pi_nodes    | `[16, 16]`      | specifies the property-interaction hidden layers                      |
| ii_nodes    | `[16, 16]`      | specifies the interaction-interaction hidden layers                   |
| out_nodes   | `[16, 16]`      | specifies the output hidden layers                                    |
| out_units   | `1`             | the dimension of outputs                                              |
| out_pool    | `False`         | `min`, `max` or `sum`, pool atomic outputs to give global predictions |
| act         | `'tanh'`        | activation function to use                                            |
| depth       | `4`             | number of graph-convolution layers to use                             |

\bibliography

# The PiNet network

The PiNet network implements the network architecture described in our
paper.[@2020_ShaoHellstroemEtAl] The network architecture features the
graph-convolution which recursively generates atomic properties from local
environment. One distinctive feature of PiNet is that the convolution operation
is realized with pairwise functions whose form are determined by the pair,
called pairwise interactions.

## Network architecture

The overall architecture of PiNet is illustrated with the illustration below:

![PiNet architecture](../tikz/pinet.svg){width="780"}

The preprocess part of the network are implemented with shared layers (see
[Layers](./layers.md)). The graph-convolution (GC) block are further divided
into PI and IP operations, each consists several layers. Those operations are
recursively applied to update the latent variables, and the output is updated
after each iteration (`OutLayer`).

We classify the latent variables into the atom-centered "properties"
($\mathbb{P}$) and the pair-wise "interactions" ($\mathbb{I}$) in our notation.
Since the layers that transform $\mathbb{P}$ to $\mathbb{P}$ or $\mathbb{I}$ to
$\mathbb{I}$ are usually standard feed-forward neural networks (`FFLayer`), the
special part of PiNet are `PILayer` and `IPLayers`, which transform between
those two types of variables.

We use the superscript to identify each tensor, and the subscripts to
differentiate the indices of different types for each variable, following the
convention:

- $b$: basis function index;
- $\alpha,\beta,\gamma,\ldots$: feature channels;
- $i,j,k,\ldots$: atom indices;
- $x,y,z$: Cartesian coordinate indices.

$\mathbb{P}^{t}_{i\alpha}$ thus denote value of the $\alpha$-th channel of the
$i$-th atom in the tensor $\mathbb{P}^{t}$. We always provide all the subscripts
of a given tensor in the equations below, so that the dimensionality of each
tensor is unambiguously implied.

For instance, $r_{ij}$ entails a scalar distance defined between
each pair of atoms, indexed by $i,j$; $\mathbb{P}_{i\alpha}$ entails the atomic
feature vectors indexed by $i$ for the atom, and $\alpha$ for the channel. The
equations that explain each of the above layers and the hyperparameters
available for the PiNet network are detailed below.

The parameters for `PiNet` are outlined in the network specification and can be applied in the configuration file as shown in the following snippet:

```
"network": {
    "name": "PiNet",
    "params": {
        "atom_types": [1, 8],
        "basis_type": "gaussian",
        "depth": 5,
        "ii_nodes": [16, 16, 16, 16],
        "n_basis": 10,
        "out_nodes": [16],
        "pi_nodes": [16],
        "pp_nodes": [16, 16, 16, 16],
        "rc": 6.0,
    }
},
```

## Network specification

### pinet.PiNet

::: pinn.networks.pinet.PiNet

## Layer specifications

### pinet.FFLayer

::: pinn.networks.pinet.FFLayer

### pinet.PILayer

::: pinn.networks.pinet.PILayer

### pinet.IPLayer

::: pinn.networks.pinet.IPLayer

### pinet.ResUpdate

::: pinn.networks.pinet.ResUpdate

### pinet.OutLayer

::: pinn.networks.pinet.OutLayer

\bibliography

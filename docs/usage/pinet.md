# The PiNet network

The PiNet network implements the network architecture described in our
paper.[@2020_ShaoHellstroemEtAl] The network architecture features the
graph-convolution which recursively generates atomic properties from local
environment. One distinctive feature of PiNet is that the convolution operation
is realized with pairwise functions whose form are determined by the pair,
called pairwise interactions.

## Network architecture

The overall architecture of PiNet is illustrated with the illustration below:

![PiNet architecture](../tikz/pinet.svg){width="750"}

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

We use the subscripts to denote the dimensionality of each variable, following
the convention:

- $b$: basis function index;
- $c,d,e$: feature channels;
- $i,j,k$: atom indices;
- $x,y,z$: Cartesian coordinate indices.

The equations that explain each of the above layers and the hyperparameters
available for the PiNet network are detailed below.

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

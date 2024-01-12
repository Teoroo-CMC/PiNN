# The PiNet2 network

PiNet2 represents the next generation of PiNet, now equipped with equivariant support. This network architecture incorporates graph convolution to iteratively derive atomic scalar and tensor properties from the local environment. One noteworthy aspect of PiNet2 is its utilization of convolution operations that are executed through pairwise functions, whose specific forms are dictated by the pairs themselves, known as pairwise interactions, while also maintaining equivariant features.

## Network architecture

The overall architecture of PiNet is illustrated with the illustration below:

![PiNet architecture](../tikz/pinet2.svg){width="750"}

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

To 

We use the subscripts to denote the dimensionality of each variable, following
the convention:

- $b$: basis function index;
- $c,d,e$: feature channels;
- $i,j,k$: atom indices;
- $x,y,z$: Cartesian coordinate indices.

The equations that explain each of the above layers and the hyperparameters
available for the PiNet network are detailed below.

## Network specification

### pinet2.PiNet2

::: pinn.networks.pinet2.PiNet2

## Layer specifications

### pinet2.FFLayer

::: pinn.networks.pinet2.FFLayer

### pinet2.PILayer

::: pinn.networks.pinet2.PILayer

### pinet2.PIXLayer

::: pinn.networks.pinet2.PIXLayer

### pinet2.DotLayer

::: pinn.networks.pinet2.DotLayer

### pinet2.ScaleLayer

::: pinn.networks.pinet2.ScaleLayer

### pinet2.IPLayer

::: pinn.networks.pinet2.IPLayer

### pinet2.ResUpdate

::: pinn.networks.pinet2.ResUpdate

### pinet2.OutLayer

::: pinn.networks.pinet2.OutLayer

\bibliography

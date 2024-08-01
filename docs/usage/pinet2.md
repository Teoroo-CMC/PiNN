# The PiNet2 network

PiNet2 represents the next generation of PiNet, now equipped with equivariant support. This network architecture incorporates graph convolution to iteratively derive atomic scalar and tensor properties from the local environment. One noteworthy aspect of PiNet2 is its utilization of convolution operations that are executed through pairwise functions, whose specific forms are dictated by the pairs themselves, known as pairwise interactions, while also maintaining equivariant features.

Update: The new modularized PiNet2 supports scalar, vector, and tensor representations. Maximum rank can be specified by using `rank` argument at initialization. Intermediate variables also can be transformed and exposed by using `out_extra`. `out_extra={'p3': 1}` indicates that, in addition to the scalar output, a dictionary will be returned containing the key `p3` with a `Tensor` value shaped as `(..., n_channel=1)`.

Those new arguments can also be specified in input yaml file. The new model now named `NewPiNet2`(will be changed before merge into master branch). A snippet looks like:
```
network:
  name: NewPiNet2
  params:
    depth: 5
    rc: 4.5
    weighted: False
    rank: 5
    out_extra:
        - p3: 1
      ...
```
NOTE: only a customized `Model` can work with `out_extra` since it breaks function signature and return two variables. 

## Network architecture

The overall architecture of PiNet2 is illustrated with the illustration below:

![PiNet2 architecture](../tikz/pinet2.svg){width="750"}

PiNet2 follows the structure of pinet and adds first-order equivarible, which are demonstrated in blue nodes. Equalvariance-target layers are implemented and tested, e.g. `PIXLayer`, `ScaleLayer` and `DotLayer`, and rest of layers reuse PiNet code. The details about those layer can be found below. 

Indices denoted the dimensionality of each variable still following previous the convention:

- $b$: basis function index;
- $\alpha,\beta,\gamma,\ldots$: feature channels;
- $i,j,k,\ldots$: atom indices;
- $x,y,z$: Cartesian coordinate indices.

The number in left-top of a variable indicates the dimension. For instance, ${}^{3}\mathbb{P}^{t}_{ix\zeta}$ means it is a property in $\mathbb{R}^3$, and `x` is index representing three coordinates. 

The equations that explain each of the above layers and the hyperparameters
available for the PiNet2 network are detailed below.

## Network specification

### pinet2.PiNet2

::: pinn.networks.pinet2_modularized.PiNet2

## Layer specifications

### pinet2.PIXLayer

::: pinn.networks.pinet2_modularized.PIXLayer

### pinet2.DotLayer

::: pinn.networks.pinet2_modularized.DotLayer

### pinet2.ScaleLayer

::: pinn.networks.pinet2_modularized.ScaleLayer

\bibliography

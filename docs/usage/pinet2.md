# The PiNet2 network

PiNet2 represents the next generation of PiNet, now equipped with equivariant support. PiNet2 shows a significant and cost-effective improvement on energy and force predictions cross different types of datasets ranging from small molecules to liquid electrolytes, as compared to PiNet. The equivariant features turn out to also improve significantly the dipole and polarizability predictions, as demonstrated by the upgraded PiNet2-dipole and PiNet2-$\chi$.

The new modularized PiNet2 supports scalar, vectorial, and tensorial representations. Maximum rank can be specified by using `rank` argument at initialization. Intermediate variables also can be transformed and exposed by using `out_extra`. `out_extra={'p3': 1}` indicates that, in addition to the scalar output, a dictionary will be returned containing the key `p3` with a `Tensor` value shaped as `(..., n_channel=1)`.

## Network architecture

The overall architecture of PiNet2 is illustrated with the illustration below:

![PiNet2 architecture](../tikz/pinet2.svg){width="750"}

PiNet2 builds upon the structure of PiNet, incorporating vectorial and tensorial equivariables represented by the blue and green nodes. The invariant `P1` is implemented through the `InvarLayer`, while the equivariants `P3` and `P5` utilize the `EquivarLayer` without non-linear activations. Further details on these new layers are provided below."

Indices denoted the dimensionality of each variable still following previous the convention:

- $b$: basis function index;
- $\alpha,\beta,\gamma,\ldots$: feature channels;
- $i,j,k,\ldots$: atom indices;
- $x,y,z$: Cartesian coordinate indices.

The number in the upper left of a variable denotes its dimension. For instance, ${}^{3}\mathbb{P}^{t}_{ix\zeta}$ represents a property in $\mathbb{R}^3$, where $x$ indicates an index for the three spatial coordinates. Here, $t$ is an iterator, and $t + 1$ increments up to the total number of graph convolution (CG) blocks.

The parameters for `PiNet2` are outlined in the network specification and can be applied in the configuration file as shown in the following snippet:

```
"network": {
    "name": "PiNet2",
    "params": {
        "atom_types": [1, 8],
        "basis_type": "gaussian",
        "depth": 5,
        "ii_nodes": [16, 16, 16, 16],
        "n_basis": 10,
        "out_nodes": [16],
        "pi_nodes": [16],
        "pp_nodes": [16, 16, 16, 16],
        "rank": 3,
        "rc": 6.0,
        "weighted": False
    }
},
```

## Network specification

### pinet2.PiNet2

::: pinn.networks.pinet2.PiNet2

## Layer specifications

### pinet2.PIXLayer

::: pinn.networks.pinet2.PIXLayer

### pinet2.DotLayer

::: pinn.networks.pinet2.DotLayer

### pinet2.ScaleLayer

::: pinn.networks.pinet2.ScaleLayer

### pinet2.OutLayer

::: pinn.networks.pinet2.OutLayer

### pinet2.InvarLayer

::: pinn.networks.pinet2.InvarLayer

### pinet2.EquivarLayer

::: pinn.networks.pinet2.EquivarLayer

\bibliography

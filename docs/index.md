# Introduction

PiNN stands for pair-wise interaction neural network, a Python library built on top of TensorFlow for performing atomistic ML tasks. Besides training ML interatomic potentials, PiNN can also predict physical and chemical properties of molecules and materials (e.g. dipole moment and polarizability). It can be used together with the adaptive learn-on-the-fly plugin PiNNAcLe[@shao2024pinnacleadaptivelearnontheflyalgorithm] and the heterogeneous electrode plugin PiNNwall[@PiNNwall] for modelling electrochemical systems.

## Flexibility

![](images/implement.png){width="600"}

PiNN is built with modularized components, and we try to make it as easy as
possible. You do not have to rewrite everything if you just want to design a new
network structure, or apply an existing network to new datasets or new
properties.


## Scalability

PiNN fully adheres to TensorFlow's high-level Estimator and Dataset API. It is
straightforward to train and predict on different computing platforms (CPU,
multiple-GPU, cloud, etc.) without explicit programming.

## Examples

The quickest way to start with PiNN is to follow our example
[notebooks](notebooks/overview.md). The notebooks provide guides to train a
simple ANN potential with a public dataset, import your own data or further
customize the PiNN for your need.


## Cite PiNN

If you find PiNN useful, welcome to cite it as:

> [1] Li, J.; Knijff, L.; Zhang, Z.-Y.; Andersson, L.; Zhang, C. PiNN: Equivariant Neural Network Suite for Modelling Electrochemical Systems. J. Chem. Theory Comput., 2025, 21: 1382.

> [2] Shao, Y.; Hellström, M.; Mitev, P. D.; Knijff, L.; Zhang, C. PiNN: A Python Library for Building Atomic Neural Networks of Molecules and Materials. J. Chem. Inf. Model., 2020, 60: 1184.

??? note "Bibtex"
    ```bibtex
      @article{doi:10.1021/acs.jctc.4c01570,
      author  = {Li, Jichen and Knijff, Lisanne and Zhang, Zhan-Yun and Andersson, Linn{\'e}a and Zhang, Chao},
      journal = {J. Chem. Theory Comput.},
      title   = {{PiNN}: Equivariant Neural Network Suite for Modeling Electrochemical Systems},
      volume  = {21},
      number  = {3},
      pages   = {1382-1395},
      year    = {2025},
      doi     = {10.1021/acs.jctc.4c01570},
      URL     = {https://doi.org/10.1021/acs.jctc.4c01570}
      }
    ```
    ```bibtex
    @Article{2020_ShaoHellstroemEtAl,
      author    = {Yunqi Shao and Matti Hellström and Pavlin D. Mitev and Lisanne Knijff and Chao Zhang},
      journal   = {J. Chem. Inf. Model.},
      title     = {{PiNN}: A Python Library for Building Atomic Neural Networks of Molecules and Materials},
      year      = {2020},
      month     = {jan},
      number    = {3},
      pages     = {1184--1193},
      volume    = {60},
      doi       = {10.1021/acs.jcim.9b00994},
      publisher = {American Chemical Society ({ACS})},
    }
    ```


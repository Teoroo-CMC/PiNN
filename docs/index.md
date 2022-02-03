# Introduction

PiNN is a Python library built on top of TensorFlow for building
atomic neural networks (ANNs).  The primary usage of PiNN is to build
and train ANN interatomic potentials, but PiNN is also capable of
predicting physical and chemical properties of molecules and
materials.

## Flexibility

![](images/implement.png)

PiNN is built with modularized components, and we try to make it as
easy as possible. You do not have to rewrite everything if you just
want to design a new network structure, or apply an existing network
to new datasets or new properties.


## Scalability

PiNN fully adheres to TensorFlow's high-level Estimator and Dataset
API.  It is straightforward to train and predict on different
computing platforms (CPU, multiple-GPU, cloud, etc) without explicit
programming.

## Examples

The quickest way to start with PiNN is to follow our example
[notebooks](notebooks/overview.md). The notebooks provide guides to train a
simple ANN potential with a public dataset, import your own data or further
customize the PiNN for your need.


## Cite PiNN

If you find PiNN useful, welcome to cite it as:

> Y. Shao, M. Hellström, P. D. Mitev, L. Knijff, and C. Zhang. PiNN: a python
> library for building atomic neural networks of molecules and materials. J.
> Chem. Inf. Model., 60:1184–1193, January 2020. doi:10.1021/acs.jcim.9b00994.

??? note "Bibtex"
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

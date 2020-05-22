==============
Data structure
==============

PiNN works with atomic structures. Since datasets of molecules and materials
usually comes with different shapes, PiNN uses a customized way to reprent the
data to process them efficiently.

In PiNN, data is stored in dictionary of tensors. Some keys in the dictionary
are reserved for certain values, as listed below (the `n_struct` dimension
exists only when the tensors are batched).

======== ====================================== =======================
 Key      Description                            Shape
======== ====================================== =======================
 cell     lattice vectors                        [(n_struct), 3]
 e_data   energy labels for a stucture           [(n_struct)]
 s_data   stress tensor labels for a structure   [(n_struct), 3, 3]
 ind_1    index of the structure for a atom      [n_atoms, 1]
 elems    atomic numbers of the atoms            [n_atoms]
 coord    coordinates of the atoms               [n_atoms, 3]
 f_data   force labels for a stucture            [n_atoms, 3]
 ind_2    indice of the atoms for a pair         [n_pairs, 2]
 dist     pairwise distances                     [n_pairs]
 diff     pairwise distance vectors              [n_pairs, 3]
======== ====================================== =======================

Indices
=======

Atomic neural networks often deal with high-dimensional properties related to
atoms or atom pairs. PiNN represents these data as something like
`tf.IndexedSlices` in TensorFlow, with the difference that indices are
explicitly Tenors in the dictionary.

In PiNN, "ind_1" stands for 1-body (atomic) indices, "ind_2" stands for 2-body
(pairwise) indices, etc. PiNN uses those indices to perform operations that
generates a pairwise property (called pairwise interaction) from atomic one or
vice versa.

Most PiNN layers and models assumes that each input dictionary of tensors
contains multiple structures. Therefore, `ind_1` is a single index for each
atom, denoting the index of the corresponding structure in the "batch". While
`ind_2` are two indices for each pair, denoting the indices of the atoms with
the batch, as illustrated below.

Order of the pair
=================

It's worth mentioning the in PiNN `ind_2` is not necessarily symmetric. This
means for certain pairwise perperty I I_ij != I_ji. By convention I_ij
corresponds to [i, j] in `ind_2`, where `i` is considered as the center atom
when summing up the interaction.

Many-body terms
===============

It is possible to represent higher order interactions in PiNN. For example, an
angle can be seen as a pair of bonds, and thus represented with an `ind_3`
index. This is used to calculate angular fingerprints in networks like `BPNN`.


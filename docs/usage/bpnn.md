# The Behler-Parrinello Neural Network

## Introduction 

Behler-Parrinello Neural Network[@2007_BehlerParrinello] (BPNN) is an ANN
architecture developed by Jörg Behler and Micheler Parrinello. It features the
description of atomic environments with the so called symmetry functions (SFs)
and the usage of element specific neural network for atomic energies.

!!! Note 
    The symmetry functions (SFs) are defined according to the Behler's
    tutorial review on neural network potentials in 2015.[@2015_Behler] Note
    that the naming of symmetry functions is different in the original paper.
    [@2007_BehlerParrinello]

## Example `sf_spec`

```Python
[{'type':'G2', 'i': 1, 'j': 8, 
  'Rs': [1.,2.], 'eta': [0.1,0.2]},
 {'type':'G2', 'i': 8, 'j': 1,
  'Rs': [1.,2.], 'eta': [0.1,0.2]},
 {'type':'G4', 'i': 8, 'j': 8,
  'lambd':[0.5,1], 'zeta': [1.,2.], 'eta': [0.1,0.2]}]
```

## Preprocessing the SFs

The computation of angular BP symmetry functions is costly, especially when
training a BPNN potential with force labels. The process can be greatly
accelerated by caching the SFs and the derivative of those SFs with respect to
the input coordinates.

```Python
# Write a preprocessed 
from pinn.io import write_tfrecord 
ds = ... # load dataset here
bpnn = BPNN(**network_params)
write_tfrecord('ds_pre.yml', ds.map(bpnn.preprocess))
```

It's worth noting that the SFs can occupy large disk space. To work around this,
you can cache the pre-computed SFs in memory or (when that's not feasible) in a
scratch file with TensorFlow's dataset API.

```Python
# Caching the dataset in memory
ds_cached = ds.map(bpnn.preprocess).cache() 
# Or in a local scratch file 
ds_cached = ds.map(bpnn.preprocess).cache('/tmp/scratch') 
```



\bibliography

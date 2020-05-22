Manipulating datasets
=====================

Batching the dataset
--------------------

Most TensorFlow operations (caching, repeating, shuffling) can be
directly applied to the dataset. However, to handle datasets with
different numbers of atoms in each structure, which is often the case,
we use a special ``sparse_batch`` operation to create minibatches of
the data in a sparse form. For example:

.. code-block:: python

   from pinn.io import sparse_batch
   dataset = # Load some dataset here
   batched = dataset.apply(sparse_batch(100))
   

TFRecord
--------

The tfrecord format is a serialized format for efficient data reading in
TensorFlow. The format is especially useful for streaming the data over a
network. When writing the dataset, a .yml file records the data structure of the
dataset, and a .tfr file holds the data. For example:

.. code-block:: python

   from glob import glob
   from pinn.io import load_QM9, sparse_batch
   from pinn.io import write_tfrecord, load_tfrecord
   # Load QM9 dataset
   filelist = glob('/home/yunqi/datasets/QM9/dsgdb9nsd/*.xyz')
   train_set = load_QM9(filelist)['train'].apply(sparse_batch(10))
   # Write as tfrecord and read
   write_tfrecord('train.yml', train_set)
   dataset = load_tfrecord('train.yml')

Preprocessing datasets
----------------------

It is worth noting that the TFRecord format does not have a fixed dataset
format. This means it can store batched as preprocessed dataset. This can be
useful when one needs to reuse the dataset or the preprcessing is slow.

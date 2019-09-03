=====================
Manipulating datasets
=====================

Batching the dataset
--------------------

Most tensorflow operations (caching, repeating, shuffuling) can be
directly applied to the dataset. However, to handle dataset with
different number of atoms, which is often the case, we use a special
``sparse_batch`` operation to create the minibatches of the data in a
sparse form. For example:

.. code-block:: python

   from pinn.io import sparse_batch
   dataset = # Load some dataset here
   batched = dataset.apply(sparse_batch(100))
   
API reference
+++++++++++++

.. autofunction:: pinn.io.sparse_batch


TFRecord
--------

For larger datasets which do not fit in the memory, caching a dataset
or training over the cloud , it's more efficient to store the data in
the tfrecord format.

The tfrecord writer/loader also supports batched datasets, and
preprocessed datasets. When writing the dataset, a .yml file records
the data structure of the dataset, and a .tfr file is holds the
data. For example:

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

The training set will be saved in the ``train.tfr``, while
``train.yml`` holds the information about the data structure.

API reference
+++++++++++++

.. autofunction:: pinn.io.write_tfrecord
		  
.. autofunction:: pinn.io.load_tfrecord


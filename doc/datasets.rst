Datasets
========

PiNN use tensorflow's dataset class to handel data. Each data point is
a dictionary of tensors. The data point should at least include the
``'coord': coordinates`` and ``'elems': atomic numbers`` as its
features. For training potentials, ``'e_data': energy`` and optionally
``'f_data': forces`` are used as labels.


Reading a dataset
-----------------

Datasets can represented as a list of datums. In a simplest case, the
dataset could be a list of structure files, each contains one
structure and label (or a sample). PiNN provides a list_loader
decorator which, given a function that reads a single sample, turns it
into a function that transform a list of samples into a tfrecord
dataset.

.. code-block:: python

    from pinn.io import list_loader
    
    @list_loader()
    def load_file_list(filename):
	# read a single file here
        coord = ...
	elems = ...
	e_data = ...
	datum = {'coord': coord, 'elems':elems, 'e_data': e_data}
	return datum

An example :doc:`notebook <notebooks/3_Customized_dataset>` with more
details is also provided at on preparing datasets.

Splitting the dataset
---------------------

It is common practice to split the dataset into subsets for validation
in ML tasks. Our dataset loaders support a ``split`` option to do
this, the ``split`` can be a nested dictionary of relative ratios of
subsets, the dataset loader will return a nested structure of
datasets with corresponding ratios. For example:

.. code:: python
	  
    dataset = load_qm9(filelsit, split={'train':8, 'test':[1,2,3]}
    train = dataset['train']
    test1 = dataset['test'][0]
    
By default, the dataset are splitted into three subsets (train: 80%,
test: 10%, vali: 10%). Note that the loaders also requires a seed
parameter for the split to be consistent, which is defaulted to 0.


Batching the dataset
--------------------

Most tensorflow operations (caching, repeating, shuffuling) can be
directly applied to the dataset. However, to handle dataset with
different number of atoms, which is often the case, we use a special
``sparse_batch`` operation to create the minibatches of the data in a
sparse form.

API reference
+++++++++++++

.. autofunction:: pinn.io.sparse_batch

Example usage
+++++++++++++

.. code-block:: python

   from pinn.io import sparse_batch
   dataset = # Load some dataset here
   batched = dataset.apply(sparse_batch(100))

TFRecord
--------

For larger datasets which do not fit in the memory, caching a dataset
or training over the cloud , it's more efficient to store the data in
the tfrecord format.

The tfrecord writer/loader also supports batched datasets, and
preprocessed datasets. When writing the dataset, a .yml file
records the data structure of the dataset, and a .tfr file is
holds the data.

API reference
+++++++++++++
.. autofunction:: pinn.io.write_tfrecord
		  
.. autofunction:: pinn.io.load_tfrecord

Example usage
+++++++++++++

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


Implemented loaders
-------------------

For common QML datasets, we provide the function to directly load the
dataset. The loaders might be limited by IO, but if you have enough
memory, you can simply cache the dataset with ``dataset.cache()`` or
convert them to tfrecords.

The RuNNer Format
+++++++++++++++++

RuNNer data (used by the RuNNer code:
http://www.uni-goettingen.de/en/560580.html) has the format::

    begin
    lattice float float float
    lattice float float float
    lattice float float float
    atom floatcoordx floatcoordy floatcoordz int_atom_symbol floatq 0  floatforcex floatforcey floatforcez
    atom 1           2           3           4               5      6  7           8           9
    energy float
    charge float
    comment arbitrary string
    end

The order of the lines within the begin/end block are arbitrary The
coordinates are given in bohr, the charges in atomic units, the energy
in Ha, and the force components in Ha/bohr

.. autofunction:: pinn.io.load_runner

QM9 dataset
+++++++++++

The QM9 dataset includes many computed properties for 134 stable
organic molecules.  See
ref. :cite:`ramakrishnan_dral_dral_rupp_anatole_von_lilienfeld_2017`
for more details

The default behavior here is to label the internal energy "U0" as
"e_data".  This behavior can be tweaked with the :code:`label_map`
parameter.

.. autofunction:: pinn.io.load_qm9

ANI-1 dataset
+++++++++++++

The ANI-1 dataset consists of 20M off-equilibrium DFT energies for
orginic molecules.  See ref. :cite:`smith_isayev_roitberg_2017` for
more details

.. autofunction:: pinn.io.load_ani


Numpy dataset
+++++++++++++

Another easy way to generate you own dataset is to store the data as a
dictionary of numpy arrays. See how it's done in the :doc:`toy problem
<notebooks/4_Toy_LJ_with_three_atoms>`.

============
Loading data
============

Reading a dataset
-----------------

To be able to shuffle and split the dataset, we require the dataset to
be represented as a list of datums. In a simplest case, the dataset
could be a list of structure files, each contains one structure and
label (or a sample). PiNN provides a ``list_loader`` decorator which,
given a function that reads a single sample, turns it into a function
that transform a list of samples into a dataset. For example:

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

An example :doc:`notebook<../notebooks/Customizing_dataset>` with more
details is also provided at on preparing datasets.

Splitting the dataset
---------------------

It is common practice to split the dataset into subsets for validation
in ML tasks. Our dataset loaders support a ``split`` option to do
this, the ``split`` can be a nested dictionary of relative ratios of
subsets, the dataset loader will return a nested structure of
datasets with corresponding ratios. For example:

.. code:: python
	  
    dataset = load_qm9(filelist, split={'train':8, 'test':[1,2,3]}
    train = dataset['train']
    test1 = dataset['test'][0]

Here ``train`` and ``test1`` will become tf.dataset objects which can
be consumed by our models. By default, the dataset are split into
three subsets (train: 80%, test: 10%, vali: 10%). Note that the
loaders also requires a seed parameter for the split to be consistent,
which is defaulted to 0.

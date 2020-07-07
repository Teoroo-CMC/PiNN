# Data loaders

In PiNN, the dataset is represented with the TensorFlow `dataset` data stucture.
Several dataset loaders are implemented in PiNN to load data from common formats.

## Splitting the dataset

It is a common practice to split the dataset into subsets for validation in
machine learning tasks. Most of PiNN's file and dataset loaders support a
`split` option to do this. The `split` can be a nested dictionary of relative
ratios of subsets. The dataset loader will return a nested structure of datasets
with corresponding ratios. For example:

```Python
dataset = load_qm9(filelist, split={'train':8, 'test':[1,2,3]}
train = dataset['train']
test1 = dataset['test'][0]
```

Here `train` and `test1` will become tf.dataset objects which can be consumed by
our models. By default, the dataset are split into three subsets (train: 80%,
test: 10%, vali: 10%). Note that the loaders also requires a seed parameter for
the split to be consistent, and its default value is 0.


## Manipulating datasets

### Batching the dataset

Most TensorFlow operations (caching, repeating, shuffling) can be
directly applied to the dataset. However, to handle datasets with
different numbers of atoms in each structure, which is often the case,
we use a special ``sparse_batch`` operation to create minibatches of
the data in a sparse form. For example:

```Python
from pinn.io import sparse_batch
dataset = # Load some dataset here
batched = dataset.apply(sparse_batch(100))
```
   

### TFRecord

The tfrecord format is a serialized format for efficient data reading in
TensorFlow. The format is especially useful for streaming the data over a
network.

PiNN can save datasets in the TFRecord dataset. When PiNN writes the dataset, it
creates a `.yml` file records the data structure of the dataset, and a `.tfr`
file holds the data. For example:

```Python
from glob import glob
from pinn.io import load_QM9, sparse_batch
from pinn.io import write_tfrecord, load_tfrecord
# Load QM9 dataset
filelist = glob('/home/yunqi/datasets/QM9/dsgdb9nsd/*.xyz')
train_set = load_QM9(filelist)['train'].apply(sparse_batch(10))
# Write as tfrecord and read
write_tfrecord('train.yml', train_set)
dataset = load_tfrecord('train.yml')
```

It is worth noting that the TFRecord format does not have a fixed dataset
format. This means it can store batched or preprocessed dataset. This can be
useful when one needs to reuse the dataset or the preprcessing is slow.


## Reading  custom data formats

To be able to shuffle and split the dataset, PiNN require the dataset to be
represented as a list of datums. In the simplest case, the dataset could be a
list of structure files, each contains one structure and label (or a sample).
PiNN provides a `list_loader` decorator which turns a function reading a
single sample into a function that transform a list of samples into a dataset.
For example:

```Python
from pinn.io import list_loader

@list_loader()
def load_file_list(filename):
    # read a single file here
    coord = ...
    elems = ...
    e_data = ...
    datum = {'coord': coord, 'elems':elems, 'e_data': e_data}
    return datum
```

An example notebook on preparing datasets can be found
[here](../notebooks/Customizing_dataset.ipynb).

## Available data formats

| Format | Loader      | Description                                                                |
|--------|-------------|----------------------------------------------------------------------------|
| cp2k   | `load_cp2k` | CP2K output                                                                |
| QM9    | `load_qm9`  | A xyz-like file format used in the QM9[@2014_RamakrishnanDralEtAl] dataset |
| ANI    | `load_ani`  | HD5-based format used in the ANI-1[@2017_SmithIsayevEtAl] dataset          |

\bibliography

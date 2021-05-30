# Data loaders

In PiNN, the dataset is represented with the TensorFlow [`dataset`
class](https://www.tensorflow.org/api_docs/python/tf/data/Dataset). Several
dataset loaders are implemented in PiNN to load data from common formats.
Starting from v1.0, PiNN provides a canonical data loader `pinn.io.load_ds`
that handels dataset with different formats, see below for the API documentation
and available datasets.

## TFRecord

The tfrecord format is a serialized format for efficient data reading in
TensorFlow. PiNN can save datasets in the TFRecord dataset. When PiNN writes the
dataset, it creates a `.yml` file records the data structure of the dataset, and
a `.tfr` file holds the data. For example:

```Python
from glob import glob
from pinn.io import load_ds, write_tfrecord
from pinn.io import write_tfrecord
filelist = glob('/home/yunqi/datasets/QM9/dsgdb9nsd/*.xyz')
dataset = load_ds(filelist, fmt='qm9', split={'train':8, 'test':2})['train']
write_tfrecord('train.yml', train_set)
train_ds = load_ds('train.yml')
```

We advise you to convert your dataset into the TFRecord format for training. The
advantage of using this format is that it allows for the storage of preprocessed
data and batched dataset.

## Splitting the dataset

It is a common practice to split the dataset into subsets for validation in
machine learning tasks. PiNN dataset loaders support a `split` option to do
this. The `split` can be a dictionary specifying the subsets and their relative
ratios. The dataset loader will return a dictionary of datasets with
corresponding ratios. For example:

```Python
from pinn.io import load_ds
dataset = load_ds(files, fmt='qm9', split={'train':8, 'test':2})
train = dataset['train']
test = dataset['test']
```

Here `train` and `test` will become tf.dataset objects which can be consumed by
our models. The loaders also aceepts a seed parameter for the split to be
consistent, and its default value is `0`.

## Batching the dataset

Most TensorFlow operations (caching, repeating, shuffling) can be
directly applied to the dataset. However, to handle datasets with
different numbers of atoms in each structure, which is often the case,
we use a special ``sparse_batch`` operation to create minibatches of
the data in a sparse form. For example:

```Python
from pinn.io import sparse_batch
dataset = load_ds(fileanme)
batched = dataset.apply(sparse_batch(100))
```

## Custom format

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

An example notebook on preparing a custom dataset is
[here](../notebooks/Customizing_dataset.ipynb).

## Available formats

| Format | Loader          | Description                                                                                                        |
|--------|-----------------|--------------------------------------------------------------------------------------------------------------------|
| tfr    | `load_tfrecord` | See [TFRecord](#tfrecord)                                                                                          |
| runner | `load_runner`   | Loader for datasets in the [RuNNer](https://www.uni-goettingen.de/de/560580.html) foramt                           |
| ase    | `load_ase`      | Load the files with the [`ase.io.iead`](https://wiki.fysik.dtu.dk/ase/_modules/ase/io/formats.html#iread) function |
| qm9    | `load_qm9`      | A xyz-like file format used in the QM9[@2014_RamakrishnanDralEtAl] dataset                                         |
| ani    | `load_ani`      | HD5-based format used in the ANI-1[@2017_SmithIsayevEtAl] dataset                                                  |
| cp2k   | `load_cp2k`     | Loader for [CP2K](https://www.cp2k.org/) output (experimental)                                                     |

## API documentation

### pinn.io.load_ds
::: pinn.io:load_ds

### pinn.io.load_tfrecord
::: pinn.io:load_tfrecord

### pinn.io.load_ase
::: pinn.io:load_ase

### pinn.io.load_runner
::: pinn.io:load_runner

### pinn.io.load_qm9
::: pinn.io:load_qm9

### pinn.io.load_ani
::: pinn.io:load_ani

### pinn.io.load_cp2k
::: pinn.io:load_cp2k

\bibliography

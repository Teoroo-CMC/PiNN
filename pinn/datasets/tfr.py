""":code:`pinn.dataset.tfr` 
is a helper module to write Dataset objects as tfrecord files.

The functions expect a format_dict to specify the format of 
the dataset.
Each key of the format_dict should be a dictionary with two 
keys, a 'dtype' of tensorflow dtype and a 'shape' of integer list.

For example:

.. code-block:: python

   format_dict = {
       'atoms': {'dtype':  int_dtype,   'shape': [n_atoms]},
       'coord': {'dtype':  float_dtype, 'shape': [n_atoms, 3]},
       'e_data': {'dtype': float_dtype, 'shape': []}}
"""
import sys
import tensorflow as tf


_int64_feature = lambda value: tf.train.Feature(
    int64_list=tf.train.Int64List(value=value))
_float_feature = lambda value: tf.train.Feature(
    float_list=tf.train.FloatList(value=value))

_feature_fn = {
    tf.float32: _float_feature,
    tf.int32: _int64_feature}

_dtype_map = {tf.float32: tf.float32,
             tf.int32: tf.int64}
        
def write_tfrecord(dataset, fname, format_dict,
                   batch=0, log_every=100):
    """Helper function to convert dataset object into tfrecord file.

    The dataset can be batched for faster loading. 
    In that case, the remainder part of the dataset will be dropped.
    
    Args:
        dataset (Dataset): input dataset.
        fname (str): filename of the dataset to save.
        format_dict (dict): shape and dtype of dataset.
        batch (int): 
            write chunk of data instead of one sample per record
            defaults to 0 (no batching).
    """
    if batch is not 0:
        dataset = dataset.batch(batch, drop_remainder=True)
    writer = tf.python_io.TFRecordWriter(fname)
    items = dataset.make_one_shot_iterator().get_next()
    sess = tf.Session()
    print('Converting to {} ...'.format(fname))
    n_parsed = 0
    try:
        while True:
            example = tf.train.Example(
                features = tf.train.Features(
                    feature = {key: _feature_fn[format_dict[key]['dtype']](
                        val.flatten())
                               for key, val in sess.run(items).items()}))
            writer.write(example.SerializeToString())
            n_parsed += (1 if batch == 0 else batch)
            if n_parsed % log_every==0:
                sys.stdout.write('\r {} samples'.format(n_parsed))
                sys.stdout.flush()
    except tf.errors.OutOfRangeError:
        sys.stdout.write('\rDone, {} samples.'.format(n_parsed))
        sys.stdout.flush()
        sess.close()
        writer.close()


def load_tfrecord(fname, format_dict, batch=0,
                  interleave=False, cycle_length=4):
    """Load tfrecord dataset.

    While loading a batched dataset, 
    the dataset is iterated as how it was written.
    One can shuffle or re-batch the dataset with :code:`interleave=True`,
    please refer to the Tensorflow `documentation
    <https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave>`_
    for more details.

    Args:
       fname (str): filename of the dataset to load.
       format_dict (dict): shape and dtype of dataset.
       batch (int): batch size of the tfrecord dataset.
       interleave (bool): interleave the batched tfrecord.
       cycle_length (int): cycle length for the interleave.
    """
    feature_dict = {}
    for key, val in format_dict.items():
        dtype = val['dtype']
        shape = val['shape']
        if batch != 0:
            shape = [batch] + shape
        feature_dict[key] = tf.FixedLenFeature(shape, _dtype_map[dtype])
    parser = lambda example: tf.parse_single_example(example, feature_dict)
    converter = lambda tensors: {k: tf.cast(v, format_dict[k]['dtype'])
                                 for k, v in tensors.items()}
    dataset = tf.data.TFRecordDataset(fname).map(parser).map(converter)
    if interleave and batch!=0:
        dataset = dataset.interleave(
            lambda x: tf.data.Dataset.from_tensor_slices(x),
            cycle_length=cycle_length)
    return dataset

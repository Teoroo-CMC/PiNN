# -*- coding: utf-8 -*-
"""Helper functions to save/load datasets into tfrecords"""

import sys
import yaml
import tensorflow as tf
from tensorflow.python.lib.io.file_io import FileIO


def write_tfrecord(fname, dataset, log_every=100, pre_fn=None):
    """Helper function to convert dataset object into tfrecord file.

    fname must end with .yml or .yaml.
    the data will be written .tfr file with the same surfix.

    Args:
        dataset (Dataset): input dataset.
        fname (str): filename of the dataset to save.
    """
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    # Preperation
    tfr = '.'.join(fname.split('.')[:-1]+['tfr'])
    writer = tf.python_io.TFRecordWriter(tfr)
    tensors = dataset.make_one_shot_iterator().get_next()
    if pre_fn:
        tensors = pre_fn(tensors)
        dataset = dataset.map(pre_fn)
    types = dataset.output_types
    shapes = dataset.output_shapes
    # Sanity check
    assert (type(types) == dict and all(type(v) != dict for v in types.values())),\
        "Only dataset of non-nested dictionary is supported."
    assert fname.endswith('.yml'), "Filename must end with .yml."
    serialized = {k: tf.serialize_tensor(v) for k, v in tensors.items()}
    sess = tf.Session()
    # Writing Loop
    n_parsed = 0
    try:
        while True:
            features = {}
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={key: _bytes_feature(val)
                             for key, val in sess.run(serialized).items()}))
            writer.write(example.SerializeToString())
            n_parsed += 1
            if n_parsed % log_every == 0:
                sys.stdout.write('\r {} samples written to {} ...'
                                 .format(n_parsed, tfr))
                sys.stdout.flush()
    except tf.errors.OutOfRangeError:
        print('\r {} samples written to {}, done.'.format(n_parsed, tfr))
        sess.close()
        writer.close()
    # Write metadata
    format_dict = {k: {'dtype': types[k].name, 'shape': shapes[k].as_list()}
                   for k in types.keys()}
    info_dict = {'n_sample': n_parsed}
    with FileIO(fname, 'w') as f:
        yaml.safe_dump({'format': format_dict, 'info': info_dict}, f)


def load_tfrecord(fname):
    """Load tfrecord dataset.

    Args:
       fname (str): filename of the .yml metadata file to load.
       dtypes (dict): dtype of dataset.
    """
    # dataset
    with FileIO(fname, 'r') as f:
        format_dict = (yaml.safe_load(f)['format'])
    dtypes = {k: format_dict[k]['dtype'] for k in format_dict.keys()}
    shapes = {k: format_dict[k]['shape'] for k in format_dict.keys()}

    feature_dict = {k: tf.FixedLenFeature([], tf.string) for k in dtypes}

    def parser(example): return tf.parse_single_example(example, feature_dict)

    def converter(tensors):
        tensors = {k: tf.parse_tensor(v, dtypes[k])
                   for k, v in tensors.items()}
        [v.set_shape(shapes[k]) for k, v in tensors.items()]
        return tensors
    tfr = '.'.join(fname.split('.')[:-1]+['tfr'])
    dataset = tf.data.TFRecordDataset(tfr).map(parser).map(converter)
    return dataset

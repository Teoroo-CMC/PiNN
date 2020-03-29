# -*- coding: utf-8 -*-
"""Helper functions to save/load datasets into tfrecords"""

import sys
import yaml
import tensorflow as tf
from tensorflow.python.lib.io.file_io import FileIO


def write_tfrecord(fname, dataset, log_every=100, pre_fn=None):
    """Helper function to convert dataset object into tfrecord file.

    fname must end with .yml or .yaml.
    The data will be written in a .tfr file with the same suffix.

    Args:
        dataset (Dataset): input dataset.
        fname (str): filename of the dataset to be saved.
    """
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    # Preperation
    tfr = '.'.join(fname.split('.')[:-1]+['tfr'])
    writer = tf.io.TFRecordWriter(tfr)
    #tensors = dataset.make_one_shot_iterator().get_next()
    if pre_fn:
        dataset = dataset.map(pre_fn)

    spec = tf.data.DatasetSpec.from_value(dataset)._serialize()[0]
    # Sanity check
    assert (type(spec) == dict and all(type(v) != dict for v in spec.values())),\
        "Only dataset of non-nested dictionary is supported."
    assert fname.endswith('.yml'), "Filename must end with .yml."
    serialize = lambda tensors: {k: tf.io.serialize_tensor(v) for k, v in tensors.items()}
    dataset = dataset.map(serialize)

    # Write serialized data
    for i, tensors in enumerate(dataset):
        features = {}
        example = tf.train.Example(
            features=tf.train.Features(
                feature={key: _bytes_feature(val.numpy())
                         for key, val in tensors.items()}))
        writer.write(example.SerializeToString())
        if (i+1) % log_every == 0:
            sys.stdout.write('\r {} samples written to {} ...'
                             .format(i+1, tfr))
            sys.stdout.flush()
    print('\r {} samples written to {}, done.'.format(i+1, tfr))

    # Write metadata
    format_dict = {k: {'dtype': v.dtype.name, 'shape': v.shape.as_list()}
                   for k, v in spec.items()}
    info_dict = {'n_sample': i+1}
    with FileIO(fname, 'w') as f:
        yaml.safe_dump({'format': format_dict, 'info': info_dict}, f)


def load_tfrecord(fname):
    """Load tfrecord dataset.

    Args:
       fname (str): filename of the .yml metadata file to be loaded.
       dtypes (dict): dtype of dataset.
    """
    # dataset
    with FileIO(fname, 'r') as f:
        format_dict = (yaml.safe_load(f)['format'])
    dtypes = {k: format_dict[k]['dtype'] for k in format_dict.keys()}
    shapes = {k: format_dict[k]['shape'] for k in format_dict.keys()}

    feature_dict = {k: tf.io.FixedLenFeature([], tf.string) for k in dtypes}

    def parser(example): return tf.io.parse_single_example(example, feature_dict)

    def converter(tensors):
        tensors = {k: tf.io.parse_tensor(v, dtypes[k])
                   for k, v in tensors.items()}
        [v.set_shape(shapes[k]) for k, v in tensors.items()]
        return tensors
    tfr = '.'.join(fname.split('.')[:-1]+['tfr'])
    dataset = tf.data.TFRecordDataset(tfr).map(parser).map(converter)
    return dataset

"""Helper functions to save/load datasets into tfrecords"""
import sys, yaml
import tensorflow as tf

def write_tfrecord(fname, dataset, log_every=100):
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
    # Sanity check
    types = dataset.output_types
    shapes = dataset.output_shapes
    assert (type(types)==dict and all(type(v)!=dict for v in types.values())),\
           "Only dataset of non-nested dictionary is supported."
    assert fname.endswith('.yml'), "Filename must end with .yml."
    # Writing Loop
    tfr = '.'.join(fname.split('.')[:-1]+['tfr'])
    writer = tf.python_io.TFRecordWriter(tfr)
    tensors = dataset.make_one_shot_iterator().get_next()
    serialized = {k: tf.serialize_tensor(v) for k, v in tensors.items()}
    sess = tf.Session()
    n_parsed = 0
    try:
        while True:
            features = {}
            example = tf.train.Example(
                features = tf.train.Features(
                    feature = {key: _bytes_feature(val)
                               for key, val in sess.run(serialized).items()}))
            writer.write(example.SerializeToString())
            n_parsed += 1
            if n_parsed % log_every==0:
                sys.stdout.write(f'\r {n_parsed} samples written to {tfr} ...')
                sys.stdout.flush()
    except tf.errors.OutOfRangeError:
        print(f'\r {n_parsed} samples written to {tfr}, done.')
        sess.close()
        writer.close()
    # Write metadata
    format_dict = {k: {'dtype': types[k].name, 'shape': shapes[k].as_list()}
                   for k in types.keys()}
    info_dict = {'n_sample': n_parsed}
    with open(fname, 'w') as f:
        yaml.safe_dump({'format': format_dict, 'info': info_dict}, f)
        
def load_tfrecord(fname):
    """Load tfrecord dataset.

    Args:
       fname (str): filename of the .yml metadata file to load.
       dtypes (dict): dtype of dataset.
    """
    # dataset
    with open(fname) as f:
        format_dict = (yaml.safe_load(f)['format'])
    dtypes = {k: format_dict[k]['dtype'] for k in format_dict.keys()}
    shapes = {k: format_dict[k]['shape'] for k in format_dict.keys()}
    
    feature_dict = {k: tf.FixedLenFeature([], tf.string) for k in dtypes}
    parser = lambda example: tf.parse_single_example(example, feature_dict)
    def converter(tensors):
        tensors = {k: tf.parse_tensor(v, dtypes[k]) for k, v in tensors.items()}
        [v.set_shape(shapes[k]) for k, v in tensors.items()]
        return tensors
    tfr = '.'.join(fname.split('.')[:-1]+['tfr'])
    dataset = tf.data.TFRecordDataset(tfr).map(parser).map(converter)
    return dataset

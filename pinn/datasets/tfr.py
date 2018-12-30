"""TFRecord Datasets for PiNN

TODO: finish this and implement converting functions
"""

import tensorflow as tf


_int64_feature = lambda value: tf.train.Feature(
    int64_list=tf.train.Int64List(value=[value]))
_bytes_feature = lambda value: tf.train.Feature(
    bytes_list=tf.train.BytesList(value=[value]))
float_feature = lambda value: tf.train.Feature(
    float_list=tf.train.FloatList(value=[value]))


def write_block_tfrecord(dataset, fname,
                         format_dict,
                         chunksize=100):
    """Write dataset into a block format tfrecord,

    The function assumes that the dataset is already padded
    and each chunk will have the same size (except for the last one).
    However, the load_block_tfrecord function can work with
    heterogeneous blocks
    """
    d = dataset.batch(chunksize)
    sess = tf.Session()
    writer = tf.python_io.TFRecordWriter(fname)
    n = d.make_one_shot_iterator().get_next()
    print('Start converting, to {} ...'.format(fname))
    try:
        while True:
            example = tf.train.Example(
                features = tf.train.Features(
                    feature = {
                        'chunksize': _int64_feature(chunksize),
                        'max_atom': _int64_feature(max_atom),
                        'pos_raw': _bytes_feature(pos_raw)}))
            writer.write(example.SerializeToString())
    except tf.errors.OutOfRangeError:
        print('Convertion done, {} chunks.'.format(i))
        sess.close()
        writer.close()


def load_block_tfrecord(fname, cycle_length=32):
    """Load
    Args:
       cycle_length (int): number of parallel processes to decode the data
    """
    def _record_parser(tensors, features, shapes):
        tensors = tf.parse_single_example(tensors_tf, features)
        outputs = {}

        insert_tensor = lambda shape: [tensors[s] if s.is_string else s
                                  for s in shape]
        shapes = {k: insert_tensor(v) for k,v in shapes.items()}

        for key in (shapes):
            if key.endswith('_raw'):
                real_key = key[:-4]
                temp = tf.decode_raw(tensors[key], tf.float32)
                outputs[real_key] = tf.reshape(temp, shapes[real_key])
            elif shapes[key] is not []:
                outputs[key] = tf.reshape(tensors[key], shapes[key])
        return tf.data.Dataset.from_tensor_slices(pos)

    metadata = load(fname)
    d = tf.data.TFRecordDataset(metadata['file'])

    d = d.interleave(_record_parser, cycle_length=cycle_length)
    return d

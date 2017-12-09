from tensorflow.contrib.learn import datasets
from tensorflow import estimator
from tensorflow import python_io
import tensorflow as tf
import numpy as np


def gen_input_fn_csv(filename, num_epochs=1, shuffle=False, target_dtype=np.uint8, features_dtype=np.float32):
    data = datasets.base.load_csv_with_header(
        filename=filename,
        target_dtype=target_dtype,
        features_dtype=features_dtype)

    x = np.array(data.data)
    x = (x - np.mean(x)) / np.std(x)

    return estimator.inputs.numpy_input_fn(
        x={'x': x},
        y=np.array(data.target),
        num_epochs=num_epochs,
        shuffle=shuffle)


def gen_input_fn_tfrecords(filename, batch_size=64, shuffle_size=None, repeat=1):
    def input_fn():
        dataset = tf.contrib.data.TFRecordDataset([filename])
        dataset = dataset.map(parse_function)
        if shuffle_size is not None:
            dataset = dataset.shuffle(shuffle_size)
        dataset = dataset.repeat(repeat)
        dataset = dataset.batch(batch_size)
        features, label = dataset.make_one_shot_iterator().get_next()
        return features, label
    return input_fn


def list_to_tfrecord(input_file, tfrecord_filename, label_col=-1):
    with python_io.TFRecordWriter(tfrecord_filename) as writer:
        with open(input_file, 'r') as lines:
            for line in lines:
                row = line.rstrip().split(',')
                features, label = row[:label_col], row[label_col]
                features = [float(f) for f in features]
                label = int(label)
                example = tf.train.Example()
                example.features.feature['x'].float_list.value.extend(features)
                example.features.feature['y'].int64_list.value.append(label)
                writer.write(example.SerializeToString())


def parse_function(example_proto):
    features = {
        'x': tf.FixedLenFeature((99 * 161,), tf.float32),
        'y': tf.FixedLenFeature((), tf.int64)
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    return parsed_features['x'], parsed_features['y']


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, help='Path to input file')
    parser.add_argument('--out_file', type=str, help='Path to output file')
    FLAGS, _ = parser.parse_known_args()

    if FLAGS.in_file is not None:
        list_to_tfrecord(FLAGS.in_file, FLAGS.out_file)

    shuffle_size = 50
    batch_size = 25


    def input_fn():
        dataset = tf.contrib.data.TFRecordDataset([FLAGS.out_file])
        dataset = dataset.map(parse_function)
        dataset = dataset.shuffle(shuffle_size)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        features, label = dataset.make_one_shot_iterator().get_next()
        return features, label


    features, labels = input_fn()
    with tf.Session() as sess:
        f_data, l_data = sess.run([features, labels])
    print(f_data, l_data)

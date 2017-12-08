from tensorflow import python_io
import tensorflow as tf
from util.specgram import from_file
from util.labels import label2int
import numpy as np
import os


def list_to_tfrecord(input_file, tfrecord_filename, label_col=-1):
    with python_io.TFRecordWriter(tfrecord_filename) as writer:
        with open(input_file, 'r') as lines:
            total_lines = os.popen('wc -l ' + input_file).read().strip()
            total_lines = total_lines.split(' ')[0]
            i = 0
            for line in lines:
                root = 'data_speech_commands_v0.01'
                key, wav = line.strip().split()
                path = root + '/' + key + '/' + wav
                spec = from_file(path)

                if spec.shape != (99, 161):
                    print('>>> bad file: {} - {}'.format(path, spec.shape))
                    continue
                features = np.asarray(spec.reshape((1, spec.shape[0] * spec.shape[1])), dtype=np.float32)
                label = label2int(key)
                example = tf.train.Example()
                example.features.feature['x'].float_list.value.extend(features[0])
                example.features.feature['y'].int64_list.value.append(label)
                writer.write(example.SerializeToString())
                if i % 100 == 0:
                    print(str(i) + '/' + str(total_lines))
                i += 1


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
    parser.add_argument('--inspect', type=str, help='Path to tfrecord to inspect')
    FLAGS, _ = parser.parse_known_args()

    if FLAGS.inspect is not None:
        i = 0
        for example in tf.python_io.tf_record_iterator(FLAGS.in_file):
            i += 1
        print(i)
    elif FLAGS.in_file is not None:
        list_to_tfrecord(FLAGS.in_file, FLAGS.out_file)

    if FLAGS.inspect is None:
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

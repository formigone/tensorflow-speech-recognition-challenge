from tensorflow import python_io
import tensorflow as tf
from util.specgram import from_file
from util.labels import label2int
import numpy as np
import os
import random


def list_to_tfrecord(input_file, tfrecord_filename, label_col=-1, is_dir=False, max_files=None):
    with python_io.TFRecordWriter(tfrecord_filename) as writer:
        if is_dir:
            # root = 'data_speech_commands_v0.01/test/audio/'
            # total_lines = 158539
            # i = 0
            for filename in os.listdir(input_file):
                # spec = from_file(root + filename)
                sound = from_file(input_file + '/' + filename, sound_only=True)

                # if spec.shape != (99, 161):
                #     print('>>> bad file: {} - {}'.format(filename, spec.shape))
                #     continue
                # features = np.asarray(spec.reshape((1, spec.shape[0] * spec.shape[1])), dtype=np.float32)
                features = sound
                label = 0
                example = tf.train.Example()
                # example.features.feature['x'].float_list.value.extend(features[0])
                example.features.feature['x'].float_list.value.extend(features)
                example.features.feature['y'].int64_list.value.append(label)
                writer.write(example.SerializeToString())
                # if i % 32 == 0:
                #     print(str(int(i / total_lines * 100)) + '%')
                # i += 1
                # if max_files is not None and max_files == i:
                #     break
        else:
            with open(input_file, 'r') as lines:
                total_lines = os.popen('wc -l ' + input_file).read().strip()
                total_lines = total_lines.split(' ')[0]
                i = 0
                root = 'data_speech_commands_v0.01'
                for line in lines:
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
    parser.add_argument('--max_files', type=int, help='Max files in file')
    parser.add_argument('--inspect', type=str, help='Only inspect tfrecord in in_file')
    parser.add_argument('--xp', type=str, help='Experimental')
    FLAGS, _ = parser.parse_known_args()

    if FLAGS.inspect is not None:
        for example in tf.python_io.tf_record_iterator(FLAGS.in_file):
            result = tf.train.Example.FromString(example)
            print(result.features.feature['tag'])
    elif FLAGS.xp == 'create':
        with python_io.TFRecordWriter('test.tfr') as writer:
            for y in range(100):
                features = np.asarray([y] + [float(random.random()) for x in range(3)], dtype=np.float32)
                print(features)
                label = y % 3
                example = tf.train.Example()
                example.features.feature['x'].float_list.value.extend(features)
                example.features.feature['y'].int64_list.value.append(label)
                writer.write(example.SerializeToString())
    elif FLAGS.xp == 'use':
        def func(features, labels, mode, params):
            dense = tf.layers.dense(features, units=10, activation=tf.nn.relu)
            logits = tf.layers.dense(dense, units=3)

            predictions = {
                'classes': tf.argmax(logits, axis=1),
                'probabilities': tf.nn.softmax(logits),
                'tags': features
            }

            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

            onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=3)
            loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            eval_metric_ops = {
                'accuracy': tf.metrics.accuracy(labels=labels, predictions=3)
            }

            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                eval_metric_ops=eval_metric_ops
            )

        model = tf.estimator.Estimator(model_fn=func)


        def parse_function(example_proto):
            features = {
                'x': tf.FixedLenFeature((4,), tf.float32),
                'y': tf.FixedLenFeature((), tf.int64),
            }
            parsed_features = tf.parse_single_example(example_proto, features)
            return parsed_features['x'], parsed_features['y']

        def input_fn():
            dataset = tf.contrib.data.TFRecordDataset(['test.tfr'])
            dataset = dataset.map(parse_function)
            # dataset = dataset.shuffle(0)
            dataset = dataset.repeat(1)
            dataset = dataset.batch(5)
            features, label = dataset.make_one_shot_iterator().get_next()
            return features, label

        model.train(input_fn=input_fn)
        preds = model.predict(input_fn=input_fn)
        ls = os.listdir('fake')
        print('Predictions:')
        for pred, fake in zip(preds, ls):
            print('Tag: {},  filename: {}'.format(pred['tags'][0], fake))
    elif FLAGS.in_file is not None:
        list_to_tfrecord(FLAGS.in_file, FLAGS.out_file, is_dir=True, max_files=FLAGS.max_files)
        print('Generated massive tfrec. BOOM!')

    if FLAGS.inspect is None and FLAGS.xp is None:
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

import argparse
import sys
import os

import numpy as np
import tensorflow as tf
from models import trivial_cnn, deep_cnn, deep_cnn2, deep_cnn3, deep_cnn4, deep_cnn5
from util.data import gen_input_fn_tfrecords
from util.labels import int2label

FLAGS = None
LEARNING_RATE = 0.0001
DROPOUT_RATE = 0.5
OUTPUT_CLASSES = 11
tf.logging.set_verbosity(tf.logging.DEBUG)


def predict(model):
    tf.logging.debug('Generating predictions...')
    predictions = model.predict(input_fn=gen_input_fn_tfrecords(FLAGS.input_file, repeat=1))
    tf.logging.debug('Got predictions')
    files = os.listdir('data_speech_commands_v0.01/test/audio')
    tf.logging.debug('Got list of files to label')
    i = 0
    with open(FLAGS.output_file, 'w+') as out_fh:
        out_fh.write('fname,label\n')
        for pred, filename in zip(predictions, files):
            label = int2label(np.argmax(pred['predictions']))
            out_fh.write('{},{}\n'.format(filename, label))
            if i % 1000 == 0:
                tf.logging.debug('file {}: {} (iteration {})'.format(filename, label, i))
            i += 1
    print('Saved predictions to {}'.format(FLAGS.output_file))


def main(args):
    model_params = {
        'learning_rate': LEARNING_RATE,
        'dropout_rate': DROPOUT_RATE,
        'output_classes': OUTPUT_CLASSES,
    }

    if FLAGS.model == 'deep':
        model_fn = deep_cnn
    elif FLAGS.model == 'deep-v2':
        model_fn = deep_cnn2
    elif FLAGS.model == 'deep-v3':
        model_fn = deep_cnn3
    elif FLAGS.model == 'deep-v4':
        model_fn = deep_cnn4
    elif FLAGS.model == 'deep-v5':
        model_fn = deep_cnn5
    else:
        model_fn = trivial_cnn

    model = tf.estimator.Estimator(model_dir=FLAGS.model_dir, model_fn=model_fn.model_fn, params=model_params)

    if FLAGS.mode == 'train':
        model.train(input_fn=gen_input_fn_tfrecords(FLAGS.input_file, batch_size=FLAGS.batch_size, repeat=10))
    elif FLAGS.mode == 'eval':
        model.evaluate(input_fn=gen_input_fn_tfrecords(FLAGS.input_file))
    elif FLAGS.mode == 'predict':
        predict(model)
    else:
        raise ValueError('Invalid mode')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, metavar='', default='train', help='Must be [train|eval|predict]')
    parser.add_argument('--model', type=str, metavar='', help='Must be [simple_cnn|vgg]')
    parser.add_argument('--model_dir', type=str, metavar='', default=None, help='Path to save the model to')
    parser.add_argument('--input_file', type=str, metavar='', help='Path to input file')
    parser.add_argument('--output_file', type=str, metavar='', help='Path to output file')
    parser.add_argument('--batch_size', type=int, metavar='', default=16, help='Only used in training')

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

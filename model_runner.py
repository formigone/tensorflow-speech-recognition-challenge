import argparse
import sys
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from models import trivial_cnn, deep_cnn, deep_cnn2, deep_cnn3, deep_cnn4
from util.data import gen_input_fn_tfrecords
from util.labels import int2label

FLAGS = None
LEARNING_RATE = 0.005
DROPOUT_RATE = 0.5
OUTPUT_CLASSES = 11
tf.logging.set_verbosity(tf.logging.DEBUG)


def predict(model):
    i = 1
    output_fh = open(FLAGS.output_file, 'w+')
    output_fh.write('fname,label\n')
    output_fh.close()
    # while True:
    #     file = FLAGS.input_file + str(i) + '.csv'
    #     print('Input file: {}'.format(file))
    #     if os.path.isfile(file):
    #         labels_list = file.replace('.csv', '-pred.csv')
    #         labels_fh = open(labels_list, 'r')
    #
    #         start = datetime.now()
    #         predictions = model.predict(input_fn=gen_input_fn_csv(file, num_epochs=1))
    #         total_time = datetime.now() - start
    #         print('  calculated predictions in {}'.format(total_time))
    #
    #         print('Printing predictions for {}...'.format(file))
    #         output_fh = open(FLAGS.output_file, 'a')
    #         for pred, label in zip(predictions, labels_fh):
    #             output_fh.write('{},{}\n'.format(label.strip(), int2label(np.argmax(pred['predictions']))))
    #         labels_fh.close()
    #         i += 1
    #     else:
    #         break
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
    else:
        model_fn = trivial_cnn

    model = tf.estimator.Estimator(model_dir=FLAGS.model_dir, model_fn=model_fn.model_fn, params=model_params)

    if FLAGS.mode == 'train':
        model.train(input_fn=gen_input_fn_tfrecords(FLAGS.input_file, repeat=10))
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

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

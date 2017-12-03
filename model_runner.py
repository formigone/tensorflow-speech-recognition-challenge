import argparse
import sys

import numpy as np
import tensorflow as tf
from models import simple_cnn, vgg
from util.data import gen_input_fn_csv
from util.labels import int2label

FLAGS = None
LEARNING_RATE = 1E-4
DROPOUT_RATE = 0.4
OUTPUT_CLASSES = 10
tf.logging.set_verbosity(tf.logging.DEBUG)


def main(args):
    model_params = {
        'learning_rate': LEARNING_RATE,
        'dropout_rate': DROPOUT_RATE,
        'output_classes': OUTPUT_CLASSES,
    }

    if FLAGS.model == 'vgg':
        model_fn = vgg
    else:
        model_fn = simple_cnn

    model = tf.estimator.Estimator(model_dir=FLAGS.model_dir, model_fn=model_fn.model_fn, params=model_params)

    if FLAGS.mode == 'train':
        for i in range(1000):
            file_num = (i % FLAGS.total_input_files) + 1
            filename = FLAGS.input_file_pattern.replace('{}', str(file_num))
            print('Input file: {}'.format(filename))
            model.train(input_fn=gen_input_fn_csv(filename, num_epochs=25, shuffle=True))
    elif FLAGS.mode == 'eval':
        eval = model.evaluate(input_fn=gen_input_fn_csv(FLAGS.input_file, num_epochs=1))
        print(eval)
    elif FLAGS.mode == 'predict':
        predictions = model.predict(input_fn=gen_input_fn_csv(FLAGS.input_file, num_epochs=1))
        for i, p in enumerate(predictions):
            print("Prediction %s: %s" % (i + 1, np.argmax(p['predictions'])))
    else:
        raise ValueError('Invalid mode')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, metavar='', required=True, help='Must be [train|eval|predict]')
    parser.add_argument('--model', type=str, metavar='', required=True, help='Must be [simple_cnn|vgg]')
    parser.add_argument('--input_file_pattern', type=str, metavar='', help='Path to input data file. Ex: my-file{}.csv')
    parser.add_argument('--total_input_files', type=int, metavar='', help='Max value to increment input file pattern')
    parser.add_argument('--input_file', type=str, metavar='', help='Path to input file')
    parser.add_argument('--model_dir', type=str, metavar='', default=None, help='Path to save the model to')

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

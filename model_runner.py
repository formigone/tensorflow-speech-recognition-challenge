import argparse
import sys

import tensorflow as tf
from models import simple_cnn
from util.data import gen_input_fn_csv

FLAGS = None
LEARNING_RATE = 0.001
tf.logging.set_verbosity(tf.logging.INFO)


def main(args):
    model_params = {
        'learning_rate': LEARNING_RATE
    }

    model = tf.estimator.Estimator(model_dir=FLAGS.model_dir, model_fn=simple_cnn.model_fn, params=model_params)

    if FLAGS.mode == 'train':
        print('--------')
        print('Training')
        print('--------')

        model.train(input_fn=gen_input_fn_csv(FLAGS.input_file, num_epochs=5))
    elif FLAGS.mode == 'validate':
        print('----------')
        print('Validating')
        print('----------')

        predictions = model.predict(input_fn=gen_input_fn_csv(FLAGS.input_file))
        for i, p in enumerate(predictions):
            print("Prediction %s: %s" % (i + 1, p["ages"]))
    else:
        raise ValueError('Invalid mode')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, metavar='', required=True, help='Must be [train|validate]')
    parser.add_argument('--input_file', type=str, metavar='', required=True, help='Path to input data CSV file')
    parser.add_argument('--testing_input_file', type=str, metavar='', default=None, help='Path to testing input data CSV file')
    parser.add_argument('--model_dir', type=str, metavar='', default=None, help='Path to save the model to')

    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.mode == 'train' and FLAGS.testing_input_file is None:
        print('Missing argument --testing_input_file. This file must be provided when --mode == train')
        exit()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

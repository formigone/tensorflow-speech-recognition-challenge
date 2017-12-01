import argparse
import sys

import numpy as np
import tensorflow as tf
from models import simple_cnn
from util.data import gen_input_fn_csv

FLAGS = None
LEARNING_RATE = 1E-6
DROPOUT_RATE = 0.4
tf.logging.set_verbosity(tf.logging.DEBUG)


def main(args):
    model_params = {
        'learning_rate': LEARNING_RATE,
        'dropout_rate': DROPOUT_RATE
    }

    model = tf.estimator.Estimator(model_dir=FLAGS.model_dir, model_fn=simple_cnn.model_fn, params=model_params)
    model.train(input_fn=gen_input_fn_csv(FLAGS.input_file, num_epochs=500, target_dtype=np.float32))

    if False and FLAGS.mode == 'train':
        for i in range(100):
            print('--------')
            print('Training')
            print('--------')
            model.train(input_fn=gen_input_fn_csv(FLAGS.input_file, num_epochs=500, target_dtype=np.float32))

            print('--------')
            print('Evaluating')
            print('--------')
            evaluation = model.evaluate(input_fn=gen_input_fn_csv(FLAGS.input_file, num_epochs=1, target_dtype=np.float32))
            print(evaluation)
    elif FLAGS.mode == 'validate':
        print('----------')
        print('Validating')
        print('----------')

        predictions = model.predict(input_fn=gen_input_fn_csv(FLAGS.input_file, num_epochs=1, target_dtype=np.float32))
        for i, p in enumerate(predictions):
            print("Prediction %s: %s" % (i + 1, p["classes"]))
    else:
        raise ValueError('Invalid mode')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, metavar='', required=True, help='Must be [train|validate]')
    parser.add_argument('--input_file', type=str, metavar='', required=True, help='Path to input data CSV file')
    parser.add_argument('--model_dir', type=str, metavar='', default=None, help='Path to save the model to')

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

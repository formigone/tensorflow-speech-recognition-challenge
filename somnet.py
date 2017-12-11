import tensorflow as tf
import argparse

FLAGS = None


def gen_parse_fn(input_shape=(16000,)):
    def parse_fn(example_proto):
        features = {
            'x': tf.FixedLenFeature(input_shape, tf.float32),
            'y': tf.FixedLenFeature((), tf.int64)
        }
        parsed_features = tf.parse_single_example(example_proto, features)
        return parsed_features['x'], parsed_features['y']

    return parse_fn


def gen_input_fn(filename, batch_size=16, shuffle_size=None, repeat=1):
    def input_fn():
        dataset = tf.contrib.data.TFRecordDataset([filename])
        dataset = dataset.map(gen_parse_fn())
        if shuffle_size is not None:
            dataset = dataset.shuffle(shuffle_size)
        dataset = dataset.repeat(repeat)
        dataset = dataset.batch(batch_size)
        features, label = dataset.make_one_shot_iterator().get_next()
        return features, label

    return input_fn


def get_kernel_weights(layer_name):
    conv1_kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, '{}/kernel'.format(layer_name))
    return tf.transpose(conv1_kernel[0], perm=[3, 0, 1, 2])


def model_fn(features, labels, mode, params):
    print('Features')
    print(features)
    tf.summary.histogram('features', features)
    tf.summary.audio('features', features, sample_rate=16000, max_outputs=6)

    feat_vol_1 = tf.reshape(features, [-1, 125, 128, 1], name='feat_vol_1')
    feat_vol_2 = tf.reshape(features, [-1, 128, 125, 1], name='feat_vol_2')
    tf.summary.image('features_1', feat_vol_1)
    tf.summary.image('features_2', feat_vol_2)

    conv1 = tf.layers.conv2d(feat_vol_1, filters=16, kernel_size=5, activation=tf.nn.relu, name='conv1')
    tf.summary.image('conv1', get_kernel_weights('conv1'), max_outputs=16)

    # features_complex = tf.cast(features, dtype=tf.complex64)
    # fft = tf.spectral.fft2d(features_complex, name='fft')
    # print(fft)
    # fft_float = tf.cast(fft, dtype=tf.float32)
    # tf.summary.image('fft', fft_float)

    dense = tf.layers.dense(conv1, units=10, activation=tf.nn.relu, name='dense')
    logits = tf.layers.dense(dense, units=params['num_classes'], name='logits')
    pred_argmax = tf.argmax(logits, axis=1, name='argmax')
    pred_softmax = tf.nn.softmax(logits, name='softmax')

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={'predictions': pred_softmax, 'classes': pred_argmax})

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=params['num_classes'], name='onehot_labels')
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
    tf.summary.scalar('loss', loss)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(labels=labels, predictions=pred_argmax)
    }
    tf.summary.scalar('accuracy', eval_metric_ops['accuracy'][1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops
    )


def main(unparsed_args):
    params = {
        'learning_rate': 1e-3,
        'num_classes': 3,
    }
    model = tf.estimator.Estimator(model_dir=FLAGS.model_dir, model_fn=model_fn, params=params)

    if FLAGS.mode == 'train':
        tf.logging.debug('Training {} on {}'.format(FLAGS.model_dir, FLAGS.input_file))
        model.train(input_fn=gen_input_fn(FLAGS.input_file))

    elif FLAGS.mode == 'predict':
        tf.logging.debug('Predicting')
        predictions = model.predict(input_fn=gen_input_fn(FLAGS.input_file))
        for pred in predictions:
            print(pred)

    else:
        raise ValueError('Invalid --mode')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train or predict')
    parser.add_argument('--model_dir', type=str, help='Path to TF checkpoints')
    parser.add_argument('--input_file', type=str, help='Path to input tf records file')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main, unparsed)

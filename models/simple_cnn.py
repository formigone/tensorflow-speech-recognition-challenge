import tensorflow as tf


def model_fn(features, labels, mode, params):
    print('labels:')
    print(labels)
    l1 = tf.layers.dense(features['x'], 15939, activation=tf.nn.relu, name='dense_1')
    output_layer = tf.layers.dense(l1, 1, name='output_layer')
    predictions = tf.reshape(output_layer, [-1])

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={'pred': predictions})

    return tf.estimator.EstimatorSpec(
        mode=mode,
    )

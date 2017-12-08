import tensorflow as tf


def model_fn(features, labels, mode, params):
    input = tf.reshape(features['x'], [-1, 99, 161, 1])
    tf.summary.image('input', input)

    # output: ((n + 2p - f) / s) + 1
    # output: n / 2

    # (99, 161, 1)
    conv1 = tf.layers.conv2d(input, filters=16, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv1')
    # (97, 159, 32)
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2, name='pool1')
    # (48, 79, 32)

    pool1_flat = tf.reshape(pool1, [-1, 48 * 79 * 16], name='pool1_flat')

    dense2 = tf.layers.dense(pool1_flat, units=128, activation=tf.nn.relu, name='dense2')

    logits = tf.layers.dense(dense2, units=params['output_classes'], name='logits')

    predictions = {
        'classes': tf.argmax(logits, axis=1, name='prediction_classes'),
        'probabilities': tf.nn.softmax(logits, name='prediction_softmax')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={'predictions': predictions['probabilities']})

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=params['output_classes'], name='onehot_labels')
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
    tf.summary.scalar('loss', loss)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions['classes'])
    }

    tf.summary.scalar('accuracy', eval_metric_ops['accuracy'][1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops
    )

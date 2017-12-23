import tensorflow as tf

def stric_block(prev, filters, mode, name, only_same=False):
    conv = prev
    with tf.variable_scope(name):
        conv = tf.layers.conv2d(conv, filters=filters, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv1')
        conv = tf.layers.conv2d(conv, filters=filters, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv2')
        if only_same:
            conv = tf.layers.conv2d(conv, filters=filters, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv3')
        else:
            conv = tf.layers.conv2d(conv, filters=filters, kernel_size=3, activation=tf.nn.relu, name='conv3')
        conv = tf.layers.conv2d(conv, filters=filters, kernel_size=3, strides=2, activation=tf.nn.relu, name='conv4')
        conv = tf.layers.batch_normalization(conv, training=mode == tf.estimator.ModeKeys.TRAIN, name='batch_norm')
    return conv


def model_fn(features, labels, mode, params):
    x = tf.reshape(features, [-1, 99, 161, 1], name='input_stric_conv2')
    x_norm = tf.layers.batch_normalization(x, training=mode == tf.estimator.ModeKeys.TRAIN, name='x_norm')
    if params['verbose_summary']:
        tf.summary.image('input', x)

    conv = x_norm
    conv = stric_block(conv, 32, mode, 'conv_1')
    conv = stric_block(conv, 64, mode, 'conv_2')
    conv = stric_block(conv, 128, mode, 'conv_3')
    conv = stric_block(conv, 256, mode, 'conv_4')
    conv = stric_block(conv, 512, mode, 'conv_5', only_same=True)

    flat = flatten(conv)
    dropout1 = tf.layers.dropout(flat, rate=params['dropout_rate'], training=mode == tf.estimator.ModeKeys.TRAIN, name='dropout1')
    dense1 = tf.layers.dense(dropout1, units=2048, activation=tf.nn.relu, name='dense1')
    dropout2 = tf.layers.dropout(dense1, rate=params['dropout_rate'], training=mode == tf.estimator.ModeKeys.TRAIN, name='dropout2')
    dense2 = tf.layers.dense(dropout2, units=2048, activation=tf.nn.relu, name='dense2')
    dropout3 = tf.layers.dropout(dense2, rate=params['dropout_rate'], training=mode == tf.estimator.ModeKeys.TRAIN, name='dropout3')

    logits = tf.layers.dense(dropout3, units=params['num_classes'], name='logits')

    predictions = {
        'classes': tf.argmax(logits, axis=1, name='prediction_classes'),
        'probabilities': tf.nn.softmax(logits, name='prediction_softmax')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={'predictions': predictions['probabilities']})

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=params['num_classes'], name='onehot_labels')
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
    tf.summary.scalar('loss', loss)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
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


def flatten(input, name='flatten'):
  dim = input.get_shape()[1:]
  dim = int(dim[0] * dim[1] * dim[2])
  return tf.reshape(input, [-1, dim], name=name)

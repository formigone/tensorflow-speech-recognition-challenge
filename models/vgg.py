import tensorflow as tf


def model_fn(features, labels, mode, params):
    '''
    Network stats:
    | layer    | output shape   | parameters
    +----------+----------------+--------------
    | conv4    | (8, 16, 512)   | 4,608
    | pool4    | (4, 8, 512)    | 0
    | flat     | (16384)        | 0
    | dense4   | (128)          | 2,097,280
    | dense5   | (64)           | 8,256
    | softmax  | (10)           | 650
    \ TOTAL    |                | 2,112,842

    :param features:
    :param labels:
    :param mode:
    :param params:
    :return:
    '''
    input = tf.reshape(features['x'], [-1, 99, 161, 1])
    tf.summary.image('input', input)

    # output: ((n + 2p - f) / s) + 1
    # output: n / 2

    # (99, 161, 1)
    conv1 = tf.layers.conv2d(input, filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv1')
    # (99, 161, 64)
    conv2 = tf.layers.conv2d(conv1, filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv2')
    # (99, 161, 64)
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2, name='pool2')
    # (49, 80, 64)
    tf.summary.image('pool2', pool2[:, :, :, 0:1])

    conv3 = tf.layers.conv2d(pool2, filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv3')
    # (49, 80, 128)
    conv4 = tf.layers.conv2d(conv3, filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv4')
    # (49, 80, 128)
    pool4 = tf.layers.max_pooling2d(conv4, pool_size=[2, 2], strides=2, name='pool4')
    # (24, 40, 128)
    tf.summary.image('pool4', pool4[:, :, :, 0:1])

    conv5 = tf.layers.conv2d(pool4, filters=256, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv5')
    # (24, 40, 256)
    conv6 = tf.layers.conv2d(conv5, filters=256, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv6')
    # (24, 40, 256)
    pool6 = tf.layers.max_pooling2d(conv6, pool_size=[2, 2], strides=2, name='pool6')
    # (12, 20, 256)
    tf.summary.image('pool6', pool6[:, :, :, 0:1])

    conv7 = tf.layers.conv2d(pool6, filters=512, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv7')
    # (12, 20, 512)
    conv8 = tf.layers.conv2d(conv7, filters=512, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv8')
    # (12, 20, 512)
    pool8 = tf.layers.max_pooling2d(conv8, pool_size=[2, 2], strides=2, name='pool8')
    # (6, 10, 512)
    tf.summary.image('pool8', pool8[:, :, :, 0:1])

    pool8_flat = tf.reshape(pool8, [-1, 6 * 10 * 512], name='pool8_flat')
    # (30720)
    dense9 = tf.layers.dense(pool8_flat, units=512, activation=tf.nn.relu, name='dense9')
    # (512)
    dense10 = tf.layers.dense(dense9, units=512, activation=tf.nn.relu, name='dense10')
    # (512)
    dropout = tf.layers.dropout(dense10, rate=params['dropout_rate'], training=mode == tf.estimator.ModeKeys.TRAIN, name='dropout')

    logits = tf.layers.dense(dropout, units=params['output_classes'], name='logits')

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
        # 'rmse': tf.metrics.root_mean_squared_error(tf.cast(labels, tf.float32), predictions),
        'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions['classes'])
    }

    tf.summary.scalar('accuracy', eval_metric_ops['accuracy'][1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops
    )

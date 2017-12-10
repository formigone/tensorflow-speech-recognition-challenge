import tensorflow as tf


def model_fn(features, labels, mode, params):
    '''
    Network stats:
    | layer    | output shape   | parameters
    +----------+----------------+--------------
    | conv1    | (99, 161, 64)  |
    | conv2    | (97, 159, 64)  |
    | conv3    | (95, 157, 64)  |
    | pool3    | (47, 78, 64)   | 0

    | conv4    | (47, 78, 128)  |
    | conv5    | (45, 76, 128)  |
    | conv6    | (43, 74, 128)  |
    | pool6    | (21, 37, 128)  | 0

    | conv7    | (21, 37, 256)  |
    | conv8    | (19, 35, 256)  |
    | conv9    | (17, 33, 256)  |
    | pool9    | (8, 16, 256)   | 0

    | conv10   | (6, 14, 512)   |
    | pool10   | (3, 7, 512)    | 0
    | flat     | (10752)        | 0
    | dense11  | (2048)         |
    | dense12  | (2048)         |
    | softmax  | (11)           |
    \ TOTAL    |                |

    :param features:
    :param labels:
    :param mode:
    :param params:
    :return:
    '''

    x = tf.reshape(features, [-1, 99, 161, 1], name='input_deep_cnn4')
    x_norm = tf.layers.batch_normalization(x, training=mode == tf.estimator.ModeKeys.TRAIN, name='x_norm')

    tf.summary.image('input', x)

    # output: ((n + 2p - f) / s) + 1
    # output: n / 2

    # (99, 161, 1)
    conv1 = tf.layers.conv2d(x_norm, filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv1')
    # (99, 161, 64)
    conv2 = tf.layers.conv2d(conv1, filters=64, kernel_size=3, activation=tf.nn.relu, name='conv2')
    # (97, 159, 64)
    conv3 = tf.layers.conv2d(conv2, filters=64, kernel_size=3, activation=tf.nn.relu, name='conv3')
    # (95, 157, 64)
    pool3 = tf.layers.max_pooling2d(conv3, pool_size=[2, 2], strides=2, name='pool3')
    # (47, 78, 64)
    tf.summary.image('pool3', pool3[:, :, :, 0:1])

    tf.summary.histogram('conv1', tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv1/kernel')[0])
    tf.summary.histogram('conv2', tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv2/kernel')[0])
    tf.summary.histogram('conv3', tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv3/kernel')[0])
    tf.summary.image('pool3', pool3[:, :, :, 0:1])

    # (47, 78, 64)
    conv4 = tf.layers.conv2d(pool3, filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv4')
    # (47, 78, 128)
    conv5 = tf.layers.conv2d(conv4, filters=128, kernel_size=3, activation=tf.nn.relu, name='conv5')
    # (45, 76, 128)
    conv6 = tf.layers.conv2d(conv5, filters=128, kernel_size=3, activation=tf.nn.relu, name='conv6')
    # (43, 74, 128)
    pool6 = tf.layers.max_pooling2d(conv6, pool_size=[2, 2], strides=2, name='pool6')
    # (21, 37, 128)
    tf.summary.image('pool6', pool6[:, :, :, 0:1])

    tf.summary.histogram('conv4', tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv4/kernel')[0])
    tf.summary.histogram('conv5', tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv5/kernel')[0])
    tf.summary.histogram('conv6', tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv6/kernel')[0])
    tf.summary.image('pool6', pool6[:, :, :, 0:1])

    # (21, 37, 128)
    conv7 = tf.layers.conv2d(pool6, filters=256, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv7')
    # (21, 37, 256)
    conv8 = tf.layers.conv2d(conv7, filters=256, kernel_size=3, activation=tf.nn.relu, name='conv8')
    # (19, 35, 256)
    conv9 = tf.layers.conv2d(conv8, filters=256, kernel_size=3, activation=tf.nn.relu, name='conv9')
    # (17, 33, 256)
    pool9 = tf.layers.max_pooling2d(conv9, pool_size=[2, 2], strides=2, name='pool9')
    # (8, 16, 256)
    tf.summary.image('pool9', pool9[:, :, :, 0:1])

    tf.summary.histogram('conv7', tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv7/kernel')[0])
    tf.summary.histogram('conv8', tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv8/kernel')[0])
    tf.summary.histogram('conv9', tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv9/kernel')[0])
    tf.summary.image('pool9', pool9[:, :, :, 0:1])

    # (8, 16, 256)
    conv10 = tf.layers.conv2d(pool9, filters=512, kernel_size=3, activation=tf.nn.relu, name='conv10')
    # (6, 14, 512)
    pool10 = tf.layers.max_pooling2d(conv10, pool_size=[2, 2], strides=2, name='pool10')
    # (3, 7, 512)
    tf.summary.image('pool10', pool10[:, :, :, 0:1])

    tf.summary.histogram('conv10', tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv10/kernel')[0])
    tf.summary.image('pool10', pool10[:, :, :, 0:1])

    pool10_flat = tf.reshape(pool10, [-1, 3 * 7 * 512], name='pool10_flat')
    # (10752)
    dropout11 = tf.layers.dropout(pool10_flat, rate=params['dropout_rate'], training=mode == tf.estimator.ModeKeys.TRAIN, name='dropout11')
    dense11 = tf.layers.dense(dropout11, units=2048, activation=tf.nn.relu, name='dense11')
    dropout12 = tf.layers.dropout(dense11, rate=params['dropout_rate'], training=mode == tf.estimator.ModeKeys.TRAIN, name='dropout12')
    dense12 = tf.layers.dense(dropout12, units=2048, activation=tf.nn.relu, name='dense12')

    logits = tf.layers.dense(dense12, units=params['output_classes'], name='logits')

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

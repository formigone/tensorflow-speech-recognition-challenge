import tensorflow as tf


def model_fn(features, labels, mode, params):
    '''
    Network stats:
    | layer    | output shape   | parameters
    +----------+----------------+--------------
    | conv1    | (97, 159, 32)  | 320
    | pool1    | (97, 159, 32)  | 0
    | conv2    | (46, 77, 64)   | 576
    | pool2    | (23, 38, 64)   | 0
    | conv3    | (21, 36, 128)  | 1,152
    | pool3    | (10, 18, 128)  | 0
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
    conv1 = tf.layers.conv2d(input, filters=64, kernel_size=3, activation=tf.nn.relu, name='conv1')
    # (97, 159, 64)
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2, name='pool1')
    # (48, 79, 64)
    conv1_kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv1/kernel')
    tf.summary.histogram('conv1', conv1_kernel[0])
    tf.summary.image('conv1', tf.transpose(conv1_kernel[0], perm=[3, 0, 1, 2]), max_outputs=64)
    tf.summary.image('pool1', pool1[:, :, :, 0:1])

    conv2 = tf.layers.conv2d(pool1, filters=128, kernel_size=3, activation=tf.nn.relu, name='conv2')
    # (46, 77, 128)
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2, name='pool2')
    # (23, 38, 128)
    conv2_kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv2/kernel')
    tf.summary.histogram('conv2', conv2_kernel[0])
    tf.summary.image('pool2', pool2[:, :, :, 0:1])

    conv3 = tf.layers.conv2d(pool2, filters=256, kernel_size=3, activation=tf.nn.relu, name='conv3')
    # (21, 36, 512)
    pool3 = tf.layers.max_pooling2d(conv3, pool_size=[2, 2], strides=2, name='pool3')
    # (10, 18, 512)
    conv3_kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv3/kernel')
    tf.summary.histogram('conv3', conv3_kernel[0])
    tf.summary.image('pool3', pool3[:, :, :, 0:1])

    conv4 = tf.layers.conv2d(pool3, filters=512, kernel_size=3, activation=tf.nn.relu, name='conv4')
    # (8, 16, 512)
    pool4 = tf.layers.max_pooling2d(conv4, pool_size=[2, 2], strides=2, name='pool4')
    # (4, 8, 512)
    conv4_kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv4/kernel')
    tf.summary.histogram('conv4', conv4_kernel[0])
    tf.summary.image('pool4', pool4[:, :, :, 0:1])

    pool4_flat = tf.reshape(pool4, [-1, 4 * 8 * 512], name='pool4_flat')
    # (16384)
    dense5 = tf.layers.dense(pool4_flat, units=1024, activation=tf.nn.relu, name='dense5')
    # (128)
    dense6 = tf.layers.dense(dense5, units=1024, activation=tf.nn.relu, name='dense6')
    # (64)
    dropout = tf.layers.dropout(dense6, rate=params['dropout_rate'], training=mode == tf.estimator.ModeKeys.TRAIN, name='dropout')

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

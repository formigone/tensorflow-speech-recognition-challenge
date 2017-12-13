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
    # input = tf.reshape(features['x'], [-1, 99, 161, 1])
    x = tf.reshape(features, [-1, 99, 161, 1], name='input_deep_cnn6')
    x_norm = tf.layers.batch_normalization(x, training=mode == tf.estimator.ModeKeys.TRAIN, name='x_norm')
    if params['verbose_summary']:
        tf.summary.image('input', x)

    incep1 = inception_block(x_norm, name='incep1')
    incep2 = inception_block(incep1, t1x1=16, t3x3=16, t5x5=16, tmp=16, name='incep2')

    conv2 = tf.layers.conv2d(incep2, filters=128, kernel_size=3, activation=tf.nn.relu, name='conv2')
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2, name='pool2')
    if params['verbose_summary']:
        conv2_kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv2/kernel')
        tf.summary.histogram('conv2', conv2_kernel[0])
        tf.summary.image('pool2', pool2[:, :, :, 0:1])

    conv3 = tf.layers.conv2d(pool2, filters=256, kernel_size=3, activation=tf.nn.relu, name='conv3')
    pool3 = tf.layers.max_pooling2d(conv3, pool_size=[2, 2], strides=2, name='pool3')
    if params['verbose_summary']:
        conv3_kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv3/kernel')
        tf.summary.histogram('conv3', conv3_kernel[0])
        tf.summary.image('pool3', pool3[:, :, :, 0:1])

    conv4 = tf.layers.conv2d(pool3, filters=512, kernel_size=3, activation=tf.nn.relu, name='conv4')
    pool4 = tf.layers.max_pooling2d(conv4, pool_size=[2, 2], strides=2, name='pool4')
    if params['verbose_summary']:
        conv4_kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv4/kernel')
        tf.summary.histogram('conv4', conv4_kernel[0])
        tf.summary.image('pool4', pool4[:, :, :, 0:1])

    dim = pool4.get_shape()[1:]
    dim = int(dim[0] * dim[1] * dim[2])
    pool4_flat = tf.reshape(pool4, [-1, dim], name='pool4_flat')
    dropout5 = tf.layers.dropout(pool4_flat, rate=params['dropout_rate'], training=mode == tf.estimator.ModeKeys.TRAIN, name='dropout5')
    dense5 = tf.layers.dense(dropout5, units=2048, activation=tf.nn.relu, name='dense5')
    dropout6 = tf.layers.dropout(dense5, rate=params['dropout_rate'], training=mode == tf.estimator.ModeKeys.TRAIN, name='dropout6')
    dense6 = tf.layers.dense(dropout6, units=2048, activation=tf.nn.relu, name='dense6')

    logits = tf.layers.dense(dense6, units=params['output_classes'], name='logits')

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


def inception_block(prev, t1x1=8, t3x3=8, t5x5=8, tmp=8, name='incep'):
    with tf.variable_scope(name):
        with tf.variable_scope('1x1_conv'):
            tower_1x1 = tf.layers.conv2d(prev,
                                         filters=t1x1,
                                         kernel_size=1,
                                         padding='same',
                                         activation=tf.nn.relu,
                                         name='1x1_conv')

        with tf.variable_scope('3x3_conv'):
            tower_3x3 = tf.layers.conv2d(prev,
                                         filters=t3x3,
                                         kernel_size=1,
                                         padding='same',
                                         activation=tf.nn.relu,
                                         name='1x1_conv')
            tower_3x3 = tf.layers.conv2d(tower_3x3,
                                         filters=t3x3,
                                         kernel_size=3,
                                         padding='same',
                                         activation=tf.nn.relu,
                                         name='3x3_conv')

        with tf.variable_scope('5x5_conv'):
            tower_5x5 = tf.layers.conv2d(prev,
                                         filters=t5x5,
                                         kernel_size=1,
                                         padding='same',
                                         activation=tf.nn.relu,
                                         name='1x1_conv')
            tower_5x5 = tf.layers.conv2d(tower_5x5,
                                         filters=t5x5,
                                         kernel_size=3,
                                         padding='same',
                                         activation=tf.nn.relu,
                                         name='3x3_conv_1')
            tower_5x5 = tf.layers.conv2d(tower_5x5,
                                         filters=t5x5,
                                         kernel_size=3,
                                         padding='same',
                                         activation=tf.nn.relu,
                                         name='3x3_conv_2')

        with tf.variable_scope('maxpool'):
            tower_mp = tf.layers.max_pooling2d(prev,
                                               pool_size=3,
                                               strides=1,
                                               padding='same',
                                               name='3x3_maxpool')
            tower_mp = tf.layers.conv2d(tower_mp,
                                         filters=tmp,
                                         kernel_size=1,
                                         padding='same',
                                         activation=tf.nn.relu,
                                         name='1x1_conv')
        return tf.concat([tower_1x1, tower_3x3, tower_5x5, tower_mp], axis=3)

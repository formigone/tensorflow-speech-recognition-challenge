import tensorflow as tf


def model_fn(features, labels, mode, params):
    x = tf.reshape(features, [-1, 99, 161, 1], name='input_incep13')
    x_norm = tf.layers.batch_normalization(x, training=mode == tf.estimator.ModeKeys.TRAIN, name='x_norm')
    if params['verbose_summary']:
        tf.summary.image('input', x)

    conv1 = tf.layers.conv2d(x_norm, filters=16, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv1')
    conv1b = tf.layers.conv2d(conv1, filters=16, kernel_size=3, activation=tf.nn.relu, name='conv1b')
    conv1c = tf.layers.conv2d(conv1b, filters=16, kernel_size=3, activation=tf.nn.relu, name='conv1c')
    pool1 = tf.layers.max_pooling2d(conv1c, pool_size=[2, 2], strides=2, name='pool1')
    if params['verbose_summary']:
        log_conv_kernel('conv1')
        log_conv_kernel('conv1b')
        tf.summary.image('pool1', pool1[:, :, :, 0:1])

    conv2 = tf.layers.conv2d(pool1, filters=32, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv2')
    conv2b = tf.layers.conv2d(conv2, filters=32, kernel_size=3, activation=tf.nn.relu, name='conv2b')
    conv2c = tf.layers.conv2d(conv2b, filters=32, kernel_size=3, activation=tf.nn.relu, name='conv2c')
    pool2 = tf.layers.max_pooling2d(conv2c, pool_size=[2, 2], strides=2, name='pool2')
    if params['verbose_summary']:
        log_conv_kernel('conv2')
        log_conv_kernel('conv2b')
        tf.summary.image('pool2', pool2[:, :, :, 0:1])

    conv3 = tf.layers.conv2d(pool2, filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv3')
    conv3b = tf.layers.conv2d(conv3, filters=64, kernel_size=3, activation=tf.nn.relu, name='conv3b')
    conv3c = tf.layers.conv2d(conv3b, filters=64, kernel_size=3, activation=tf.nn.relu, name='conv3c')
    pool3 = tf.layers.max_pooling2d(conv3c, pool_size=[2, 2], strides=2, name='pool3')
    if params['verbose_summary']:
        log_conv_kernel('conv3')
        log_conv_kernel('conv3b')
        tf.summary.image('pool3', pool3[:, :, :, 0:1])

    incep4 = inception_block(pool3, t1x1=32, t3x3=32, t5x5=32, tmp=32, name='incep4')
    incep5 = inception_block(incep4, t1x1=64, t3x3=64, t5x5=64, tmp=64, name='incep5')
    incep6 = inception_block(incep5, t1x1=128, t3x3=128, t5x5=128, tmp=128, name='incep6')
    incep7 = inception_block(incep6, t1x1=128, t3x3=128, t5x5=128, tmp=128, name='incep7')

    flat = flatten(incep7)
    dropout11 = tf.layers.dropout(flat, rate=params['dropout_rate'], training=mode == tf.estimator.ModeKeys.TRAIN, name='dropout11')
    dense11 = tf.layers.dense(dropout11, units=2048, activation=tf.nn.relu, name='dense11')
    dropout12 = tf.layers.dropout(dense11, rate=params['dropout_rate'], training=mode == tf.estimator.ModeKeys.TRAIN, name='dropout12')
    dense12 = tf.layers.dense(dropout12, units=2048, activation=tf.nn.relu, name='dense12')

    logits = tf.layers.dense(dense12, units=params['num_classes'], name='logits')

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


def log_conv_kernel(varname, prefix=''):
  conv_kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, '{}/kernel'.format(varname))
  tf.summary.histogram('{}/{}'.format(prefix, varname), conv_kernel[0])


def flatten(input, name='flatten'):
  dim = input.get_shape()[1:]
  dim = int(dim[0] * dim[1] * dim[2])
  return tf.reshape(input, [-1, dim], name=name)


def touch_incep(input, name):
  dim = input.get_shape()[1:]
  return tf.reshape(input, [-1, dim[0], dim[1], dim[2]], name=name)


def double_inception(prev, block_depth, name):
  tower = inception_block(prev, t1x1=block_depth, t3x3=block_depth, t5x5=block_depth, tmp=block_depth, name=name)
  tower_res = touch_incep(tower, '{}_res'.format(name))
  return inception_block(tower_res, t1x1=block_depth, t3x3=block_depth, t5x5=block_depth, tmp=block_depth, name='{}_2'.format(name))


def conv_group(prev, filters, name='conv_group', verbose=False):
  with tf.variable_scope(name):
    conv = tf.layers.conv2d(prev, filters=filters, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv_same')
    convb = tf.layers.conv2d(conv, filters=filters, kernel_size=3, activation=tf.nn.relu, name='convb')
    convc = tf.layers.conv2d(convb, filters=filters, kernel_size=3, activation=tf.nn.relu, name='convc')
    pool = tf.layers.max_pooling2d(convc, pool_size=[2, 2], strides=2, name='pool')
    if verbose:
      log_conv_kernel('{}/conv_same'.format(name))
      log_conv_kernel('{}/convb'.format(name))
      log_conv_kernel('{}/convc'.format(name))
      tf.summary.image('{}/pool'.format(name), pool[:, :, :, 0:1])
    return pool


def inception_block(prev, t1x1=2, t3x3=2, t5x5=2, tmp=2, name='incep', log_conv_weights=False):
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

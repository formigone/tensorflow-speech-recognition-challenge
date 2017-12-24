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
    x = tf.reshape(features, [-1, 99, 161, 1], name='input_stric_conv3')
    x_norm = tf.layers.batch_normalization(x, training=mode == tf.estimator.ModeKeys.TRAIN, name='x_norm')
    if params['verbose_summary']:
        tf.summary.image('input', x)

    conv = x_norm
    conv = stric_block(conv, 32, mode, 'conv_1')
    conv = stric_block(conv, 64, mode, 'conv_2')
    conv = stric_block(conv, 128, mode, 'conv_3')
    conv = stric_block(conv, 256, mode, 'conv_4')
    conv = stric_block(conv, 512, mode, 'conv_5', only_same=True)

    incep = inception_block(conv, t1x1=128, t3x3=128, t5x5=128, tmp=128, name='incep6')
    incep = inception_block(incep, t1x1=128, t3x3=128, t5x5=128, tmp=128, name='incep7')
    incep = inception_block(incep, t1x1=128, t3x3=128, t5x5=128, tmp=128, name='incep8')

    flat = flatten(incep)
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
  return inception_block(tower_res, t1x1=block_depth, t3x3=block_depth, t5x5=block_depth, tmp=block_depth,
                         name='{}_2'.format(name))


def conv_group(prev, filters, name='conv_group', verbose=False):
  with tf.variable_scope(name):
    conv = tf.layers.conv2d(prev, filters=filters, kernel_size=3, padding='same', activation=tf.nn.relu,
                            name='conv_same')
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

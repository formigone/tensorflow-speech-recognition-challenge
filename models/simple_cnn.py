import tensorflow as tf


def model_fn(features, labels, mode, params):
    input = tf.reshape(features['x'], [-1, 99, 161, 1])
    tf.summary.image('input', input, max_outputs=3)

    conv_1 = tf.layers.conv2d(input, filters=32, kernel_size=3, activation=tf.nn.relu, name='conv_1')
    # output: ((n + 2p - f) / s) + 1
    # tf.summary.image('conv_1', conv_1)
    pool_1 = tf.layers.max_pooling2d(conv_1, pool_size=[2, 2], strides=2, name='pool_1')
    # output: n / 2
    # tf.summary.image('pool_1', pool_1)

    conv_2 = tf.layers.conv2d(pool_1, filters=64, kernel_size=3, activation=tf.nn.relu, name='conv_2')
    pool_2 = tf.layers.max_pooling2d(conv_2, pool_size=[2, 2], strides=2, name='pool_2')
    # tf.summary.image('pool_2', pool_2)

    pool_2_flat = tf.reshape(pool_2, [-1, 23 * 38 * 64], name='pool_2_flat')
    dense_1 = tf.layers.dense(pool_2_flat, units=1024, activation=tf.nn.relu, name='dense_1')
    dropout = tf.layers.dropout(dense_1, rate=params['dropout_rate'], training=mode == tf.estimator.ModeKeys.TRAIN, name='dropout')

    logits = tf.layers.dense(dropout, units=10, name='logits')

    predictions = {
        'classes': tf.argmax(logits, axis=1, name='prediction_classes'),
        'probabilities': tf.nn.softmax(logits, name='prediction_softmax')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={'predictions': predictions})

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10, name='onehot_labels')
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
    tf.summary.scalar('loss', loss)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    eval_metric_ops = {
        # 'rmse': tf.metrics.root_mean_squared_error(tf.cast(labels, tf.float32), predictions),
        'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions['classes'])
    }

    # print(logits)
    # print(labels)
    # print(predictions['classes'])
    # print(eval_metric_ops['accuracy'])

    tf.summary.scalar('accuracy', eval_metric_ops['accuracy'][1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops
    )

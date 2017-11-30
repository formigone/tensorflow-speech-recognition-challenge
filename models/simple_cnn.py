import tensorflow as tf


def model_fn(features, labels, mode, params):
    print('-- model_fn --')
    print('params: {0}'.format(params))

    l1 = tf.layers.dense(features['x'], 1000, activation=tf.nn.relu, name='dense_1')
    l2 = tf.layers.dense(l1, 1000, activation=tf.nn.relu, name='dense_2')
    l3 = tf.layers.dense(l2, 1000, activation=tf.nn.relu, name='dense_3')
    output_layer = tf.layers.dense(l3, 1, name='output_layer')

    l1_sqr = tf.reshape(l1, [1, 320, 400, 1])
    tf.summary.image('l1_sqr', l1_sqr)
    tf.summary.audio('l1', l1, sample_rate=16000)

    predictions = tf.reshape(output_layer, [-1], name='predictions')

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={'predictions': predictions})

    loss = tf.losses.mean_squared_error(labels, predictions)
    tf.summary.scalar('loss', loss)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    eval_metric_ops = {
        'rmse': tf.metrics.root_mean_squared_error(tf.cast(labels, tf.float32), predictions)
    }

    tf.summary.tensor_summary('pred', predictions)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops
    )

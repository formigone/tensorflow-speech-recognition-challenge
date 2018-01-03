import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# path = '/Users/rsilveira/Desktop/samples/1-spec-1.png'
# img = plt.imread(path)
# z = np.zeros((125, 161))
# print(img.shape)
# print(img[:,:,3])
# print('---')
# img[:,:,-1] = z
# print(img[:,:,3])

p = tf.placeholder(tf.float32, [1, 5, 5, 2])
c = tf.layers.conv2d(p, filters=4, kernel_size=2, padding='same')
init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)
  arr = np.array([
    [
      [
        [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5],
      ],
      [
        [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5],
      ],
      [
        [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5],
      ],
      [
        [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5],
      ],
      [
        [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5],
      ],
    ]
  ])
  print(arr.shape)
  out = sess.run(c, feed_dict={p: arr})
  print(out.shape)

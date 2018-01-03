# Given some tfrecord file, list the percentage of samples for each label.
# Example:
# | x | y |
# +---+---+
# | a | 0 |
# | b | 0 |
# | c | 1 |
#
# Output:
# | Class | Total |
# +-------+-------+
# | 0     | 2     |
# | 1     | 1     |

import tensorflow as tf


classes = {}

for example in tf.python_io.tf_record_iterator('train_aug_12.tfrecords'):
  result = tf.train.Example.FromString(example)
  key = result.features.feature['y'].int64_list.value[0]
  if key not in classes:
    classes[key] = 0
  classes[key] += 1

print(classes)

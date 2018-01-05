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

from __future__ import division
import tensorflow as tf
import util.labels as lb


def graph(classes, total):
  row = {}
  for k, v in classes.iteritems():
    row[k] = '{0:.2f}%'.format(v / total * 100)
  print(row)


classes = {}
samples = 0

for example in tf.python_io.tf_record_iterator('train_large_40250.tfrecords'):
  samples += 1
  result = tf.train.Example.FromString(example)
  key = result.features.feature['y'].int64_list.value[0]
  key = lb.int2label(key)
  if key not in classes:
    classes[key] = 0
  classes[key] += 1
  if samples % 1000 == 0:
    graph(classes, samples)

print(classes)

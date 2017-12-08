import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import util.specgram as specgram


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


paths = [
    '../data_speech_commands_v0.01/left/012c8314_nohash_0.wav',
    '../data_speech_commands_v0.01/left/1df99a8a_nohash_0.wav',
    '../data_speech_commands_v0.01/left/620ff0fa_nohash_0.wav',
]

tfrec_filename = '../data.tfrecords'
writer = tf.python_io.TFRecordWriter(tfrec_filename)
specs = []

for path in paths:
    sample = specgram.from_file(path)
    specs.append(sample)
    sample_raw = sample.tostring()

    feat = tf.train.Features(features={
        'x': _bytes_feature(sample_raw),
        'y': _int64_feature(np.array([1]))
    })
    example = tf.train.Example(features=feat)
    writer.write(example.SerializeToString())

print(specs)
writer.close()

# if False:
#     spec_rect = np.fromstring(spec_str, dtype=np.float32)
#     spec_rect = spec_rect.reshape(spec.shape)
#     print(np.allclose(spec, spec_rect))

if False:
    plt.imshow(spec, cmap='seismic')
    plt.show()

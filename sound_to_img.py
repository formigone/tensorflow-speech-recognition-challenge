import matplotlib.pyplot as plt
import argparse
import numpy as np
from scipy import signal
from scipy.io import wavfile
from util.labels import label2int, int2label


def parse_data_file(file_path, offset, max, shape=(99, 161)):
    m = 0
    i = 0
    x_len = 0
    data = []
    file = open(file_path, 'r')
    for line in file:
        if i < offset:
            i += 1
            continue
        if m >= max:
            break
        key, wav = line.split()
        path = 'data_speech_commands_v0.01/' + key + '/' + wav
        sr, sound = wavfile.read(path)
        spec = log_specgram(sound, sr)
        if spec.shape != shape:
            continue
        flat = spec.reshape((1, spec.shape[0] * spec.shape[1]))
        if x_len == 0:
            x_len = flat.shape[1]
        flat = np.char.mod('%f', flat[0])
        data.append(','.join(flat) + ',' + str(label2int(key)))
        m += 1
    file.close()

    header = str(m) + ',' + str(x_len)
    return header, data


def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, _, spec = signal.spectrogram(audio,
                                        fs=sample_rate,
                                        window='hann',
                                        nperseg=nperseg,
                                        noverlap=noverlap,
                                        detrend=False)
    return np.log(spec.T.astype(np.float32) + eps)


def sample_specgram(filename):
    file = open(filename, 'r')
    for line in file:
        key, wav = line.split()
        path = 'data_speech_commands_v0.01/' + key + '/' + wav
        sr, sound = wavfile.read(path)
        spec = log_specgram(sound, sr)
        plt.imshow(spec)
        plt.show()
        print(spec.shape)
        flat = spec.reshape((1, spec.shape[0] * spec.shape[1]))
        file.close()
        return


def graph_specgrams(filename, save_dir, w, h):
    file = open(filename, 'r')
    i = 0
    for line in file:
        line = np.array([float(num) for num in line.split(',')])
        spec = np.reshape(line[:-1], (w, h))
        plt.imsave(save_dir + '/' + str(i) + '.png', spec)
        i += 1
    file.close()


# parse_data_file('training-set.text')
parser = argparse.ArgumentParser()
parser.add_argument('--input_list', type=str, metavar='', default=None, help='Path to raw list of files and labels')
parser.add_argument('--batch_size', type=int, metavar='', default=5000, help='Max inputs to parse per file')
parser.add_argument('--filename', type=str, metavar='', default=None, help='Name of output files')
parser.add_argument('--debug', type=int, metavar='', default=0, help='Either 0 or 1')
parser.add_argument('--graph', type=int, metavar='', default=0, help='Either 0 or 1')
parser.add_argument('--graph_save_dir', type=str, metavar='', default='', help='Either 0 or 1')
FLAGS, unparsed = parser.parse_known_args()

if FLAGS.debug:
    sample_specgram(FLAGS.input_list)
elif FLAGS.graph:
    graph_specgrams(FLAGS.input_list, FLAGS.graph_save_dir, 99, 161)
else:
    offset = 0
    i = 1
    while True:
        header, data = parse_data_file(FLAGS.input_list, offset, FLAGS.batch_size)
        data_len = len(data)
        offset += data_len
        filename = FLAGS.filename + str(i) + '.csv'
        fh = open(filename, 'w')
        fh.write(header + '\n')
        fh.write('\n'.join(data) + '\n')
        fh.close()
        print('Created {} - {} lines'.format(filename, len(data)))
        i += 1
        if data_len < FLAGS.batch_size:
            break

import argparse
import numpy as np
from skimage import io


def label2int(key):
    if key == 'car':
        return 0
    elif key == 'pizza':
        return 1
    return 2


def parse_data_file(file_path, shape=(99, 161)):
    m = 0
    i = 0
    x_len = 0
    data = []
    file = open(file_path, 'r')
    for line in file:
        key, img = line.split()
        path = 'data_speech_commands_v0.01/__img__/' + img
        img = io.imread(path, as_grey=True)
        flat = img.reshape((1, img.shape[0] * img.shape[1]))
        if x_len == 0:
            x_len = flat.shape[1]
        flat = np.char.mod('%f', flat[0])
        data.append(','.join(flat) + ',' + str(label2int(key)))
        m += 1
    file.close()

    header = str(m) + ',' + str(x_len)
    return header, data


parser = argparse.ArgumentParser()
parser.add_argument('--input_list', type=str, metavar='', default=None, help='Path to raw list of files and labels')
parser.add_argument('--filename', type=str, metavar='', default=None, help='Name of output files')
FLAGS, unparsed = parser.parse_known_args()

header, data = parse_data_file(FLAGS.input_list)
data_len = len(data)
filename = FLAGS.filename
fh = open(filename, 'w')
fh.write(header + '\n')
fh.write('\n'.join(data) + '\n')
fh.close()
print('Created {} - {} lines'.format(filename, len(data)))

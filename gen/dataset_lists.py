import os
import argparse
from sklearn.utils import shuffle


def list_classes(path, max_per_class=2500):
    files = {}
    for dir in os.listdir(path):
        if not os.path.isdir(path + '/' + dir):
            continue
        for file in os.listdir(path + '/' + dir):
            _class = labels.label2int(dir)
            _label = labels.int2label(_class)
            if _label not in files:
                # files[_label] = 0
                files[_label] = []
            files[_label].append(file)
            # files[_label] += 1

    with open('train_raw_12.txt', 'w+') as train_out:
        with open('test_raw_12.txt', 'w+') as test_out:
            for key in files.keys():
                files[key] = shuffle(files[key])
                total_samples = len(files[key])
                test_len = 200
                if key == 'noise':
                    test_len = 40
                train_len = min(total_samples - test_len, max_per_class)
                test_samples = files[key][0:test_len]
                train_samples = files[key][-train_len:]
                print('{} => {}/{}'.format(key, len(test_samples), len(train_samples)))
                label = key
                if key == 'noise':
                    label = 'silence'
                for train in train_samples:
                    _label = labels.label2int(label, v2=True)
                    train_out.write('{} {} {}\n'.format(label, train, _label))
                for test in test_samples:
                    _label = labels.label2int(label, v2=True)
                    test_out.write('{} {} {}\n'.format(label, test, _label))


if __name__ == '__main__':
    from sys import path
    from os.path import dirname as dir

    path.append(dir(path[0]))

    from util import labels

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help='Root directory to parse')

    FLAGS, _ = parser.parse_known_args()

    list_classes(FLAGS.dir)

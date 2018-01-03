import os
import argparse


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
            files[_label].append((dir, file))
            # files[_label] += 1

    test_len = 200
    validation_len = 100
    max_train_len = 2067
    with open('train_raw.txt', 'w+') as train_out:
        with open('test_raw.txt', 'w+') as test_out:
            with open('validation_raw.txt', 'w+') as validation_out:
                for key in files.keys():
                    if key == 'noise':
                        test_len = 40
                    test_samples = files[key][0:test_len]
                    validation_samples = files[key][test_len:test_len + validation_len]
                    train_samples = files[key][test_len + validation_len:]
                    if len(train_samples) > max_train_len:
                        train_samples = train_samples[0:max_train_len]
                    print('{} => {}/{}/{}'.format(key, len(train_samples), len(validation_samples), len(test_samples)))
                    label = key
                    if key == 'noise':
                        label = 'silence'
                    for dir, train in train_samples:
                        _label = labels.label2int(label, v2=True)
                        train_out.write('{} {} {} {}\n'.format(label, train, _label, dir))
                    for dir, validation in validation_samples:
                        _label = labels.label2int(label, v2=True)
                        validation_out.write('{} {} {} {}\n'.format(label, validation, _label, dir))
                    for dir, test in test_samples:
                        _label = labels.label2int(label, v2=True)
                        test_out.write('{} {} {} {}\n'.format(label, test, _label, dir))


if __name__ == '__main__':
    from sys import path
    from os.path import dirname as dir

    path.append(dir(path[0]))

    from util import labels

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help='Root directory to parse')

    FLAGS, _ = parser.parse_known_args()
    list_classes(FLAGS.dir)

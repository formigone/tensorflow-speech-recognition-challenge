import tensorflow as tf
import numpy as np


def parse_file(filename):
    x = []
    y = []
    file = open(filename, 'r')
    for line in file:
        line = line.split(',')
        x.append(line[1:])
        y.append(line[0])

        if len(y) > 99:
            break
    file.close()

    return x, y


def gen_input_fn(filename, num_epochs=1, shuffle=False):
    X, Y = parse_file(filename)

    X = np.array([[float(val) for val in line] for line in X])
    Y = np.array([int(val) for val in Y])

    return tf.estimator.inputs.numpy_input_fn(
        x=np.array(X),
        y=np.array(Y),
        num_epochs=num_epochs,
        shuffle=shuffle
    )


if __name__ == '__main__':
    print(gen_input_fn('sample.csv'))

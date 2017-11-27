# import tensorflow as tf
from tensorflow.contrib.learn import datasets
from tensorflow import estimator
import numpy as np


# def parse_file(filename):
#     x = []
#     y = []
#     file = open(filename, 'r')
#     for line in file:
#         line = line.split(',')
#         x.append(line[1:])
#         y.append(line[0])
#
#         if len(y) > 1:
#             break
#     file.close()
#
#     return x, y
#
#
# def gen_input_fn(filename, num_epochs=1, shuffle=False):
#     X, Y = parse_file(filename)
#
#     X = np.array([[float(val) for val in line] for line in X])
#     Y = np.array([int(val) for val in Y])
#
#     print('gen_input_fn')
#     # print(X)
#     print(X.shape)
#     # print(Y)
#     print(Y.shape)
#
#     return estimator.inputs.numpy_input_fn(
#         x={'x': X},
#         y=Y,
#         num_epochs=num_epochs,
#         shuffle=shuffle
#     )


def gen_input_fn_csv(filename, num_epochs=1, shuffle=False):
    data = datasets.base.load_csv_with_header(
        filename=filename,
        target_dtype=np.uint8,
        features_dtype=np.float32)

    print('data')
    print(data.data.shape)
    print(data.target.shape)
    print('---')
    return estimator.inputs.numpy_input_fn(
        x={'x': data.data},
        y=data.target,
        num_epochs=num_epochs,
        shuffle=shuffle)


if __name__ == '__main__':
    # print(gen_input_fn('../validation-set.csv'))
    print(gen_input_fn_csv('../training-set-2.csv'))

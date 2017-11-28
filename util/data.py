from tensorflow.contrib.learn import datasets
from tensorflow import estimator
import numpy as np


def gen_input_fn_csv(filename, num_epochs=1, shuffle=False, target_dtype=np.uint8, features_dtype=np.float32):
    data = datasets.base.load_csv_with_header(
        filename=filename,
        target_dtype=target_dtype,
        features_dtype=features_dtype)

    x = np.array(data.data)
    x = (x - np.mean(x)) / np.std(x)

    return estimator.inputs.numpy_input_fn(
        x={'x': x},
        y=np.array(data.target),
        num_epochs=num_epochs,
        shuffle=shuffle)


if __name__ == '__main__':
    # print(gen_input_fn('../validation-set.csv'))
    print(gen_input_fn_csv('../training-set-simple.csv'))

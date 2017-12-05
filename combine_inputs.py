import argparse
import os

FLAGS = None


def combine(files, out):
    print(files)
    print(out)
    with open(out + '.tmp', 'w+') as fh_out:
        for file in files:
            print('file: {}'.format(file))
            with open(file, 'r') as fh:
                for line in fh:
                    print(line)
                    break
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_pattern', type=str, help='Pattern of csv files to be combined')
    parser.add_argument('--dir', type=str, help='Directory where file_pattern is located')
    parser.add_argument('--out', type=str, help='Name of output file with combined data')

    FLAGS, unused = parser.parse_known_args()
    print(FLAGS.files)
#    combine(FLAGS.files.split(','), FLAGS.out)

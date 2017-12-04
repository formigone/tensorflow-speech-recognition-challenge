import argparse
import os
import random


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, metavar='', default=None, help='Path to directory to list')
parser.add_argument('--mix', type=str, metavar='', default=None, help='List of dirs to mix files from')
parser.add_argument('--max_per_mix', type=int, metavar='', default=None, help='Max files per dir to add')
parser.add_argument('--out_files', type=str, metavar='', default=None, help='List of output files (a_90, a_10)')
FLAGS, unparsed = parser.parse_known_args()

if FLAGS.mix and FLAGS.max_per_mix:
    file_splits = [[], []]
    base = 'data_speech_commands_v0.01/'
    dirs = FLAGS.mix.split(',')
    for dir in dirs:
        files = os.listdir(base + dir)
        # Collect sub-dir separately so it can be shuffled + split
        out_2 = []
        for i in range(min(len(files), FLAGS.max_per_mix)):
            out_2.append('{} {}'.format(dir, files[i]))
        random.shuffle(out_2)

        # Split 90/10
        split_point = int(len(out_2) * 0.9)
        for i in range(len(out_2)):
            if i < split_point:
                file_splits[0].append(out_2[i])
            else:
                file_splits[1].append(out_2[i])
        random.shuffle(file_splits[0])
        random.shuffle(file_splits[1])

    for i, out_file in enumerate(FLAGS.out_files.split(',')):
        out_fh = open(out_file, 'w+')
        for row in file_splits[i]:
            out_fh.write(row + '\n')
        out_fh.close()
else:
    dir = os.listdir(FLAGS.dir)
    for file in dir:
        print('test/audio {}'.format(file))

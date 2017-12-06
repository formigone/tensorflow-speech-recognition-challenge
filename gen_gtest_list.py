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
    l90 = []
    l10 = []
    base = 'data_speech_commands_v0.01/'
    dirs = FLAGS.mix.split(',')
    for dir in dirs:
        print('dir: {}'.format(dir))
        files = os.listdir(base + dir)
        # Collect sub-dir separately so it can be shuffled + split
        out_2 = []
        for i in range(min(len(files), FLAGS.max_per_mix)):
            out_2.append('{} {}'.format(dir, files[i]))
        random.shuffle(out_2)

        l90 += out_2[0:-80]
        l10 += out_2[-80:]
        random.shuffle(l90)
        random.shuffle(l10)

    out1, out2 = FLAGS.out_files.split(',')
    print('OUT: {}'.format(out1))
    with open(out1, 'w+') as fh:
        fh.write('\n'.join(l90))
    print('OUT: {}'.format(out2))
    with open(out2, 'w+') as fh:
        fh.write('\n'.join(l10))
else:
    dir = os.listdir(FLAGS.dir)
    for file in dir:
        print('test/audio {}'.format(file))

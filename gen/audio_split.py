import os
import argparse
import numpy as np


def split(filename, out_dir, len):
    basename = '.'.join(os.path.basename(FLAGS.in_file).split('.')[:-1])
    sound = specgram.load_audio_file(filename, raw=True)
    sound_len = sound.shape[0]
    i = 0
    while len * i < sound_len - len:
        offset = i * len
        chunk = sound[offset:offset + len]
        if chunk.shape[0] == len:
            synth.save_to_file('{}/{}_{}.wav'.format(out_dir, basename, i), chunk)
        i += 1
        if i % 100 == 0:
            print('{} => {}'.format(i, sound_len))


if __name__ == '__main__':
    from sys import path
    from os.path import dirname as dir

    path.append(dir(path[0]))

    from util import specgram, synth

    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, required=True, help='Path to file to be split')
    parser.add_argument('--out_dir', type=str, required=True, help='Path to dir where chunks will be stored')
    parser.add_argument('--segment_len', type=int, default=16000, help='Duration of each segment')

    FLAGS, _ = parser.parse_known_args()
    split(FLAGS.in_file, FLAGS.out_dir, FLAGS.segment_len)

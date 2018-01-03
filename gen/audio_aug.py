import numpy as np
import os
import time
import librosa
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf
from tensorflow import python_io
from scipy.io import wavfile


def load_audio_file(file_path, sr=16000):
  data = librosa.core.load(file_path, sr=sr)
  data = data[0]
  return normalize(data, sr)


def filter_speed(sound, factor):
  indices = np.round(np.arange(0, len(sound), factor))
  indices = indices[indices < len(sound)].astype(int)
  return sound[indices.astype(int)]


def remove_silence(audio, threshold):
  # identify all samples with an absolute value greater than the threshold
  greater_index = np.greater(np.absolute(audio), threshold)
  # filter to only include the identified samples
  above_threshold_data = audio[greater_index]
  return above_threshold_data


def norm_wave_len(wave, min_samples=None, max_samples=None):
  if min_samples is not None and len(wave) < min_samples:
    len_to_add = min_samples - len(wave)
    wave = np.pad(wave, (len_to_add + 1) // 2, 'median')[:min_samples]

  if max_samples is not None and len(wave) > max_samples:
    len_to_cut = len(wave) - max_samples
    wave = wave[len_to_cut // 2:max_samples + len_to_cut // 2]

  return wave


def filter_delay(data, sr, delay_rate=None):
  data1 = filter_speed(data, 0.25)
  data1 = norm_wave_len(data1, 4000, 4000)
  data = np.concatenate((data1[:4000], data))
  data = norm_wave_len(data, sr, sr)

  if delay_rate is not None:
    return filter_slow(data, sr, delay_rate)
  return data


def filter_slow(data, sr, rate=1.0):
  data = filter_speed(data, rate)
  return norm_wave_len(data, sr, sr)


def filter_pop(data, sr, rate=1.0):
  data = data / (2. ** 9)
  data = filter_speed(data, 1.1)
  data = norm_wave_len(data, sr, sr)
  data = filter_speed(data, rate)
  return norm_wave_len(data, sr, sr)


def normalize(data, sr):
  if len(data) > sr:
    data = data[:sr]
  else:
    data = np.pad(data, (0, max(0, sr - len(data))), 'constant')

  return data


def filter_stretch(data, rate=1.0, sr=16000):
  data = librosa.effects.time_stretch(data, rate)
  return normalize(data, sr)


def filter_white_noise(sound, amount=0.001):
  wn = np.random.randn(len(sound))
  return sound + amount * wn


def filter_quiet(sound, amount):
  return sound + amount * sound


def filter_roll(sound, amount=800):
  return np.roll(sound, amount)


def save_to_file(filename, data, sr=16000, out_format='wav'):
  if out_format == 'wav':
    wavfile.write(filename, sr, data)
  elif out_format == 'img':
    spec = specgram.log_specgram(data, sr)
    plt.imsave(filename, spec, cmap='gray')


def to_tfrecord(writer, x, y, sr=16000):
  x = specgram.normalize(x, sr)
  x = stacks(x, sr)
  x = x.reshape(-1)

  features = np.asarray(x, dtype=np.float32)
  label = y
  example = tf.train.Example()
  example.features.feature['x'].float_list.value.extend(features)
  example.features.feature['y'].int64_list.value.append(label)
  writer.write(example.SerializeToString())


def stacks(freqs, sr, out_shape=(125, 161)):
  spec = specgram.log_specgram(freqs, sr)
  result = np.full(out_shape, np.min(spec))
  result[:spec.shape[0], :spec.shape[1]] = spec
  spec = result

  wav = freqs.reshape((125, 128))
  result = np.full(out_shape, np.max(wav))
  result[:wav.shape[0], :wav.shape[1]] = wav
  wav = result

  z = np.zeros((125, 161, 2))
  z[:, :, 0] = spec
  z[:, :, 1] = wav

  return z


def gen_tf_record(input_list, output_file, input_dir='.', sr=16000, no_aug=False):
  total = 0

  with python_io.TFRecordWriter(output_file) as writer:
    with open(input_list, 'r') as fh:
      for line in fh:
        key, filename, label, dir = line.strip().split(' ')
        label = int(label)
        # output_path = filename.replace('.wav', '')
        file = input_dir + '/' + dir + '/' + filename
        if not os.path.isfile(file):
          continue

        data = load_audio_file(file)
        to_tfrecord(writer, data, label)
        # save_to_file(output_path + '-org.wav', data)
        # save_to_file(output_path + '-org.png', data, out_format='img')

        if not no_aug:
          for amount in [0.001, 0.005, 0.01]:
            wn = filter_white_noise(data, amount=amount)
            to_tfrecord(writer, wn, label)
            # save_to_file(output_path + '-wn.wav', wn)
            # save_to_file(output_path + '-wn.png', wn, out_format='img')

          wn2 = filter_white_noise(np.zeros(data.shape))
          wn2 = filter_slow(wn2, sr, 0.25)
          wn2 = data + wn2
          to_tfrecord(writer, wn2, label)
          # save_to_file(output_path + '-wn2.wav', wn2)
          # save_to_file(output_path + '-wn2.png', wn2, out_format='img')

          quiet = filter_quiet(data, amount=-0.5)
          quiet = filter_quiet(quiet, amount=-0.5)
          to_tfrecord(writer, quiet, label)
          # save_to_file(output_path + '-quiet.wav', quiet)
          # save_to_file(output_path + '-quiet.png', quiet, out_format='img')

          loud = filter_quiet(data, amount=108)
          to_tfrecord(writer, loud, label)
          # save_to_file(output_path + '-loud.wav', loud)
          # save_to_file(output_path + '-loud.png', loud, out_format='img')

          roll = filter_roll(data, amount=2000)
          to_tfrecord(writer, roll, label)
          # save_to_file(output_path + '-roll.wav', roll)
          # save_to_file(output_path + '-roll.png', roll, out_format='img')

          wn2 = filter_white_noise(np.zeros(data.shape))
          wn2 = filter_slow(wn2, sr, 0.5)
          roll2 = filter_roll(data, amount=2000) + wn2
          to_tfrecord(writer, roll2, label)
          # save_to_file(output_path + '-roll2.wav', roll2)
          # save_to_file(output_path + '-roll2.png', roll2, out_format='img')

          short = filter_slow(data, sr, 1.15)
          short = filter_stretch(data, rate=0.9) + short * 0.002
          to_tfrecord(writer, short, label)
          # save_to_file(output_path + '-short.wav', short)
          # save_to_file(output_path + '-short.png', short, out_format='img')

          for rate in [0.7, 0.8, 0.9]:
            deep = filter_slow(data, sr, rate=rate)
            to_tfrecord(writer, deep, label)
            # save_to_file(output_path + '-deep.wav', deep)
            # save_to_file(output_path + '-deep.png', deep, out_format='img')

          for rate in [0.7, 0.8, 0.9]:
            long = filter_stretch(data, rate=rate)
            to_tfrecord(writer, long, label)
            # save_to_file(output_path + '-long.wav', long)
            # save_to_file(output_path + '-long.png', long, out_format='img')
        if total % 250 == 0:
          now = time.asctime(time.localtime(time.time()))
          print('{} => {}  {}/{}/{}  => {}'.format(now, total, input_dir, dir, filename, label))
        total += 1


if __name__ == '__main__':
  from sys import path
  from os.path import dirname as dir

  path.append(dir(path[0]))
  import util.specgram as specgram

  parser = argparse.ArgumentParser()
  parser.add_argument('--input_file', type=str, help='List of <label:str> <filename:str> <class:int>')
  parser.add_argument('--input_dir', type=str, help='Root directory where input files are')
  parser.add_argument('--output_file', type=str, help='Path to output TFRecord file')
  parser.add_argument('--no_aug', type=bool, default=False, help='No data augmentation per file')
  FLAGS, _ = parser.parse_known_args()

  gen_tf_record(FLAGS.input_file, FLAGS.output_file, FLAGS.input_dir, no_aug=FLAGS.no_aug)

import numpy as np
import librosa
import matplotlib.pyplot as plt
import os
from scipy.io import wavfile
from util.labels import label2int
import util.specgram as specgram


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


def filter_white_noise(sound, amount=0.002):
    wn = np.random.randn(len(sound))
    return sound + amount * wn


def filter_roll(sound, amount=800):
    return np.roll(sound, amount)


def save_to_file(filename, data, sr=16000, out_format='wav'):
    if out_format == 'wav':
        wavfile.write(filename, sr, data)
    elif out_format == 'img':
        spec = specgram.log_specgram(data, sr)
        plt.imsave(filename, spec, cmap='gray')


if __name__ == '__main__':
    input_list = '../input_train_shuff'
    input_dir = '../data_speech_commands_v0.01'
    output_dir = '../data_synth'
    sr = 16000
    total = 0

    with open(input_list, 'r') as fh:
        for line in fh:
            label, path = line.strip().split(' ')
            output_path = output_dir + '/' + label
            if not os.path.isdir(output_path):
                os.mkdir(output_path)
            output_path = output_dir + '/' + label + '/' + path.replace('.wav', '')
            print(output_path)
            data = load_audio_file(input_dir + '/' + label + '/' + path)

            wn = filter_white_noise(data)
            save_to_file(output_path + '-wn.wav', wn)

            roll = filter_roll(data, amount=2000)
            save_to_file(output_path + '-roll.wav', roll)

            short = filter_slow(data, sr, 1.15)
            short = filter_stretch(data, rate=0.9) + short * 0.002
            save_to_file(output_path + '-short.wav', short)

            deep = filter_slow(data, sr, 0.85)
            save_to_file(output_path + '-deep.wav', deep)

            long = filter_stretch(data, rate=0.8)
            save_to_file(output_path + '-long.wav', long)

            if total % 50 == 0:
                save_to_file(output_path + '-long.png', long, out_format='img')
                save_to_file(output_path + '-deep.png', deep, out_format='img')
                save_to_file(output_path + '-short.png', short, out_format='img')
                save_to_file(output_path + '-wn.png', wn, out_format='img')
                save_to_file(output_path + '-roll.png', roll, out_format='img')
            total += 1


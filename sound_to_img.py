import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile


def parse_data_file(file_path, classes):
    file = open(file_path, 'r')
    for line in file:
        key, wav = line.split()
        if not classes.get(key):
            classes[key] = len(classes)
        path = 'data_speech_commands_v0.01/' + key + '/' + wav
        sr, sound = wavfile.read(path)
        spec = log_specgram(sound, sr)
        print(spec.shape)
        flat = spec.reshape((1, spec.shape[0] * spec.shape[1]))
        print(flat.shape)
        print(str(classes[key]) + ',' + ','.join(np.char.mod('%f', flat[0])))
    file.close()


def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, _, spec = signal.spectrogram(audio,
                                        fs=sample_rate,
                                        window='hann',
                                        nperseg=nperseg,
                                        noverlap=noverlap,
                                        detrend=False)
    return np.log(spec.T.astype(np.float32) + eps)


classes = {
    'up': 0,
    'down': 1,
    'left': 2,
    'right': 3,
    'go': 4,
    'stop': 5,
    'yes': 6,
    'no': 7,
    'on': 8,
    'off': 9
}

parse_data_file('validation-set.text', classes)

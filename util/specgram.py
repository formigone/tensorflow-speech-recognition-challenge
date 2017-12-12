import numpy as np
import librosa
from scipy import signal
from scipy.io import wavfile


def load_audio_file(file_path, sr=16000, raw=False):
    data = librosa.core.load(file_path, sr=sr)
    data = data[0]
    if raw:
        return data
    return normalize(data, sr)


def normalize(data, sr):
    if len(data) > sr:
        data = data[:sr]
    else:
        data = np.pad(data, (0, max(0, sr - len(data))), 'constant')
    return data


def from_file(path, sr=16000, sound_only=False):
    sound = load_audio_file(path)
    if sound_only:
        return sound
    return log_specgram(sound, sr)


def log_specgram(audio, sample_rate=16000, window_size=20, step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, _, spec = signal.spectrogram(audio,
                                        fs=sample_rate,
                                        window='hann',
                                        nperseg=nperseg,
                                        noverlap=noverlap,
                                        detrend=False)
    return np.log(spec.T.astype(np.float32) + eps)

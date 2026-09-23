"""log-mel featurizer shared by train/infer. env: dj"""
import numpy as np
import librosa
from model import SR, N_FFT, HOP, N_MELS, FMIN, FMAX, FPS


def logmel(y=None, path=None, sr=SR):
    if y is None:
        y, _ = librosa.load(path, sr=sr, mono=True)
    m = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP,
                                       n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=1.0)
    return np.log1p(m).astype("float32")        # [N_MELS, T]


def normalize(M, mu=None, sd=None):
    if mu is None:
        mu, sd = M.mean(), M.std() + 1e-6
    return (M - mu) / sd, mu, sd


def t_to_frame(t):
    return int(round(t * FPS))

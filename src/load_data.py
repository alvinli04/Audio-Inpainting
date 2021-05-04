import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from torch import nn

import io
import os
import math
import tarfile
import multiprocessing

import scipy
import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import spectrogram as sp
import model

def load_training_data():
    waveform, sample_rate = librosa.load('../data/sample.wav', sr=None)

    mspec = sp.get_mel_spectrogram(waveform, sample_rate)
    height, length = mspec.shape

    training_data = []
    for s in [i * 128 for i in range(length // 128) if i < length]:

        context = mspec[: , s : s + 96].copy()
        context = np.pad(context, ((0,0), (0, 32)), 'constant')
        target = mspec[ : , s + 96 : s + 128].copy()

        training_data.append((context, target))

        # sp.plot_mel_spectrogram(context, sample_rate)
        # sp.plot_mel_spectrogram(target, sample_rate)

    return (training_data, sample_rate)

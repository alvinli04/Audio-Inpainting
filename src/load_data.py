import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from torch import nn

import io
import os
import math
import glob
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

def load_data_file(path):
    waveform, sample_rate = librosa.load(path, sr=None)

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

def load_training_data(directory):
    files = glob.glob(directory + '**/*.flac', recursive = True)

    training_data = []
    _ , sample_rate = load_data_file(files[0])
    for file in files:
        data, sr = load_data_file(file)
        print(len(data))

        training_data += data

    return (training_data, sample_rate)

def main():
    data, sr = load_training_data('../data/')
    print(len(data))

    sp.plot_mel_spectrogram(data[0][0], sr)

if __name__ == '__main__':
    main()

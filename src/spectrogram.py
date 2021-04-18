import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

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

def plot_waveform(samples, sample_rate):
    librosa.display.waveplot(samples, sr=sample_rate)
    plt.show()

def get_mel_spectrogram(samples, sample_rate):
    sgram = librosa.stft(samples)
    sgram_mag, _ = librosa.magphase(sgram)
    mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate)
    mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
    return mel_sgram

def plot_mel_spectrogram(mel_sgram, sample_rate):
    librosa.display.specshow(mel_sgram, sr=sample_rate, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.show()

if __name__ == '__main__':
    waveform, sample_rate = librosa.load('../data/sample.wav', sr=None)
    plot_mel_spectrogram(get_mel_spectrogram(waveform, sample_rate), sample_rate)

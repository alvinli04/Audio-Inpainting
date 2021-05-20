import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

import io
import os
import math

import scipy
import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import soundfile as sf

def plot_waveform(samples, sample_rate):
    librosa.display.waveplot(samples, sr=sample_rate)
    plt.show()

def get_mel_spectrogram(samples, sample_rate):
    sgram = librosa.stft(samples, n_fft=512)
    #mess around with nfft, hopsize (match with mel_to_audio), increase mel bins
    sgram_mag, _ = librosa.magphase(sgram)
    mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate)
    mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
    return mel_sgram

def plot_mel_spectrogram(mel_sgram, sample_rate):
    librosa.display.specshow(mel_sgram, sr=sample_rate, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.show()

def get_waveform(mel_sgram, sample_rate):
    mel_sgram = librosa.db_to_amplitude(S_db=mel_sgram, ref=.000001)
    return librosa.feature.inverse.mel_to_audio(M=mel_sgram, sr=sample_rate, n_fft=512, hop_length=128)

if __name__ == '__main__':
    waveform, sample_rate = librosa.load('../data/sounds/sample.wav')
    sg = get_mel_spectrogram(waveform, sample_rate)
    wv = get_waveform(sg, sample_rate)
    plot_mel_spectrogram(sg, sample_rate)
    plot_waveform(waveform, sample_rate)
    plot_waveform(wv, sample_rate)
    sf.write('../data/sample_out.wav', wv, sample_rate, subtype='PCM_24')
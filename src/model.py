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

import spectrogram as sp


waveform, sample_rate = librosa.load('../data/sample.wav', sr=None)
mspec = sp.get_mel_spectrogram(waveform, sample_rate)
sp.plot_mel_spectrogram(mspec, sample_rate)
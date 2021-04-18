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


def main():
    waveform, sample_rate = librosa.load('../data/sample.wav', sr=None)
    mspec = sp.get_mel_spectrogram(waveform, sample_rate)
    mspec = mspec[None, None, :, :128]


    #print(mspec.shape)
    #print(mspec)
    x = torch.from_numpy(mspec)

    #print(x.size())

    m = model.Model()

    # print(m)
    #
    # params = list(m.parameters())
    # print(len(params))
    # print(params[0].size())
    m.eval()
    print(m(x))

if __name__ == "__main__":
    main()

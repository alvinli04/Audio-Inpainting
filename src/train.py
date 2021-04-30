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
import load_data as ld
import model


def main():

    training_data = ld.load_training_data()

    for context, target in training_data:
        x = torch.from_numpy(context[None, None, :, :])


        m = model.Model()
        m.eval()

        output = m(x)
        print(output)
        print(output.shape)


if __name__ == "__main__":
    main()

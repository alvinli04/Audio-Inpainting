import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from torch import nn
from torch.utils.data import Dataset, DataLoader

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


class AudioInpaintingDataset(Dataset):

    def __init__(self, root_dir, start = 0, end = None):
        self.root_dir = root_dir
        self.context_files = glob.glob(root_dir + '**/*_context.npy', recursive = True)[start : end]
        self.target_files = glob.glob(root_dir + '**/*_target.npy', recursive = True)[start : end]

        self.context_files.sort()
        self.target_files.sort()


    def __len__(self):
        return len(self.context_files)

    def __getitem__(self, idx):
        context = torch.from_numpy(np.load(self.context_files[idx])[None, None, :, :])
        target = torch.from_numpy(np.load(self.target_files[idx])[None, None, :, :])

        return (context, target)

    def get_sample_rate(self):
        fin = open(self.root_dir + 'sample_rate.txt', 'r')
        sample_rate = int(fin.read())
        fin.close()

        return sample_rate



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

def generate_training_data(directory):
    files = glob.glob(directory + '**/*.flac', recursive = True)

    training_data = []
    _ , sample_rate = load_data_file(files[0])
    for file in files:
        data, sr = load_data_file(file)
        print(len(data))

        training_data += data

    return (training_data, sample_rate)

def save_training_data(src_dir, dst_dir):
    training_data, sample_rate = generate_training_data(src_dir)
    print(len(training_data))

    for i, pair in enumerate(training_data):
        context = pair[0]
        target = pair[1]

        path = '{n}_{type}.npy'
        np.save(dst_dir + path.format(n = i, type = 'context'), context)
        np.save(dst_dir + path.format(n = i, type = 'target'), target)

    fout = open(dst_dir + 'sample_rate.txt', 'w+')
    fout.write(str(sample_rate))
    fout.close()


def main():
    save_training_data('../data/LibriSpeech/dev-clean/84/', '../data/npy_data/')

    set = AudioInpaintingDataset('../data/npy_data/',end = 3)
    loader = DataLoader(set, batch_size = 2)

    for i in range(10):
        for sample in loader:
            print(sample['context'].size())
            print(sample['target'].size())

    print(set.get_sample_rate())

if __name__ == '__main__':
    main()

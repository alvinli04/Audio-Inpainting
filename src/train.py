import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim

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
    cnn = model.Model()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(cnn.parameters(), lr=.01)


    if torch.cuda.is_available():
        cnn = cnn.cuda()
        criterion = criterion.cuda()



    train_losses, val_losses = [], []
    train_data = ld.AudioInpaintingDataset('../data/npy_data/', start = 10, end = 20)
    val_data = ld.AudioInpaintingDataset('../data/npy_data/', end = 10)
    sample_rate = train_data.get_sample_rate()

    epochs = 10
    train_loader = DataLoader(train_data)
    val_loader = DataLoader(val_data)
    print("Starting training")
    for epoch in range(epochs):

        cnn.eval()
        for contexts, targets in train_loader:

            for context, target in zip(contexts, targets):

                #training begins here
                x_train = context
                y_train = target

                x_train = Variable(x_train, requires_grad=True)
                y_train = Variable(y_train, requires_grad=True)

                if torch.cuda.is_available():
                    x_train = x_train.cuda()
                    y_train = y_train.cuda()

                out_train = cnn(x_train)
                loss_train = criterion(out_train, y_train)

                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()
                train_losses.append(loss_train.item())

        cnn.eval()
        loss = 0
        counter = 0
        for contexts, targets in val_loader:

            for context, target in zip(contexts, targets):

                x_val = context
                y_val = target

                x_val = Variable(x_val, requires_grad=True)
                y_val= Variable(y_val, requires_grad=True)

                if torch.cuda.is_available():
                    x_val = x_val.cuda()
                    y_val = y_val.cuda()

                out_val = cnn(x_val)
                loss_val = criterion(out_val, y_val)

                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()

                loss += loss_val.item()
                counter += 1

        print(loss/counter)


    test_context, test_target = val_data[-1]

    sp.plot_mel_spectrogram(np.squeeze(test_context.numpy()), sample_rate)
    sp.plot_mel_spectrogram(np.squeeze(test_target.numpy()), sample_rate)


    test_output = cnn(test_context).detach()
    sp.plot_mel_spectrogram(np.squeeze(test_output.numpy()), sample_rate)

    plt.scatter(range(len(train_losses)), train_losses)
    plt.show()


if __name__ == '__main__':
    main()

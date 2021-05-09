import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from torch import nn
from torch.autograd import Variable
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
    cnn.train()
    cnn.eval()

    train_losses, val_losses = [], []
    data, sample_rate = ld.load_training_data('../data/')

    epochs = 10
    for epoch in range(epochs):

        half = len(data) // 2
        train_data = data[ : half]
        val_data = data[half : ]

        for i in range(half):

            #training begins here
            x_train = train_data[i][0][None, None, :, :]
            y_train = train_data[i][1][None, None, :, :]

            x_val = val_data[i][0][None, None, :, :]
            y_val = val_data[i][1][None, None, :, :]

            x_train = torch.from_numpy(x_train)
            y_train = torch.from_numpy(y_train)

            x_val = torch.from_numpy(x_val)
            y_val = torch.from_numpy(y_val)

            # print(x_train.size(), y_train.size())

            # train(cnn, optimizer, criterion, x_train, y_train, x_val, y_val, train_losses, val_losses)
            x_train, y_train = Variable(x_train, requires_grad=True), Variable(y_train, requires_grad=True)
            x_val, y_val = Variable(x_val, requires_grad=True), Variable(y_val, requires_grad=True)

            if torch.cuda.is_available():
                x_train = x_train.cuda()
                y_train = y_train.cuda()
                x_val = x_val.cuda()
                y_val = y_val.cuda()

            # print(x_train.size(), y_train.size())

            out_train = cnn(x_train)
            # out_val = cnn(x_val)

            loss_train = criterion(out_train, y_train)
            # loss_val = criterion(out_val, y_val)

            optimizer.zero_grad()

            loss_train.backward()
            optimizer.step()

            train_losses.append(loss_train.item())
            # val_losses.append(loss_val)

            #training ends

        print(train_losses[-1])

    sp.plot_mel_spectrogram(val_data[-1][0], sample_rate)
    sp.plot_mel_spectrogram(val_data[-1][1], sample_rate)

    t = val_data[-1][0][None, None, :, :]
    t = torch.from_numpy(t)

    test = cnn(t).detach().numpy()[0, 0]
    sp.plot_mel_spectrogram(test, sample_rate)

    plt.scatter(range(len(train_losses)), train_losses)
    plt.show()



if __name__ == '__main__':
    main()

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

#TODO: load the test and validation sets and put them in
def train(cnn, optimizer, criterion, x_train, y_train, x_val, y_val, train_losses, val_losses):
    x_train, y_train = Variable(x_train), Variable(y_train)
    x_val, y_val = Variable(x_val), Variable(y_val)

    if torch.cuda.is_available():
        x_train = x_train.cuda()
        y_train = y_train.cuda()
        x_val = x_val.cuda()
        y_val = y_val.cuda()

    cnn.eval()
    optimizer.zero_grad()

    print(x_train.size(), y_train.size())

    out_train = cnn(x_train)
    out_val = cnn(x_val)

    loss_train = criterion(out_train, y_train)
    loss_val = criterion(out_val, y_val)

    loss_train.backward()
    optimizer.step()

    train_losses.append(loss_train)
    val_losses.append(loss_val)

def main():
    cnn = model.Model()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(cnn.parameters(), lr=.01)

    if torch.cuda.is_available():
        cnn = cnn.cuda()
        criterion = criterion.cuda()

    train_losses, val_losses = [], []

    epochs = 20
    for epoch in range(epochs):

        data = ld.load_training_data()
        half = len(data) // 2
        train_data = data[ : half]
        val_data = data[half : ]

        for i in range(half):

            x_train = train_data[i][0][None, None, :, :]
            y_train = train_data[i][1][None, None, :, :]

            x_val = val_data[i][0][None, None, :, :]
            y_val = val_data[i][1][None, None, :, :]

            x_train = torch.from_numpy(x_train)
            y_train = torch.from_numpy(y_train)

            x_val = torch.from_numpy(x_val)
            y_val = torch.from_numpy(y_val)

            # print(x_train.size(), y_train.size())

            train(cnn, optimizer, criterion, x_train, y_train, x_val, y_val, train_losses, val_losses)


if __name__ == '__main__':
    main()

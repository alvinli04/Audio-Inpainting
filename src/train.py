import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from torch import nn
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
import model

#TODO: load the test and validation sets and put them in
def train(model, optimizer, x_train, y_train, x_val, y_val, train_losses, val_losses):
    x_train, y_train = Variable(x_train), Variable(y_train)
    x_val, y_val = Variable(x_val), Variable(y_val)

    if torch.cuda.is_available():
        x_train = x_train.cuda()
        y_train = y_train.cuda()
        x_val = x_val.cuda()
        y_val = y_val.cuda()

    optimizer.zero_grad()

    out_train = model(x_train)
    out_val = model(x_val)

    loss_train = criterion(out_train, y_train)
    loss_val = criterion(out_val, y_val)

    loss_train.backward()
    optimizer.step()

    train_losses.append(loss_train)
    val_losses.append(loss_val)

def main():
    model = Model()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters, lr=.01)

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    train_losses, val_losses = [], []

    epochs = 20
    for epoch in range(epochs):
        train(model, optimizer, x_train, y_train. x_val, y_val, train_losses, val_losses)


if __name__ == '__main__':
    main()
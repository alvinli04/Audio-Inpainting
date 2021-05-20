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

PATH = '../data/models/checkpoint.tar'

def main():
    cnn = model.Model()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(cnn.parameters(), lr=.001)

    if os.path.exists(PATH):
        print("Found checkpoint")
        checkpoint = torch.load(PATH)
        cnn.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if torch.cuda.is_available():
        cnn = cnn.cuda()
        criterion = criterion.cuda()


    val_losses = []
    train_data = ld.AudioInpaintingDataset('../data/npy_data/', start = 10)
    val_data = ld.AudioInpaintingDataset('../data/npy_data/', end = 10)
    sample_rate = train_data.get_sample_rate()

    epochs = 10
    train_loader = DataLoader(train_data, batch_size = 8, drop_last = True)
    val_loader = DataLoader(val_data, batch_size = 1, drop_last = True)
    print("Starting training")
    for epoch in range(epochs):

        cnn.train()
        for context, target in train_loader:

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

        loss = 0
        counter = 0
        cnn.eval()
        for context, target in val_loader:

            x_val = context
            y_val = target

            x_val = Variable(x_val, requires_grad=True)
            y_val= Variable(y_val, requires_grad=True)

            if torch.cuda.is_available():
                x_val = x_val.cuda()
                y_val = y_val.cuda()

            out_val = cnn(x_val)
            loss_val = criterion(out_val, y_val)
            loss += loss_val.item()
            counter += 1

        print(loss/counter)

    torch.save({
            'model_state_dict': cnn.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, PATH)



    test_context, test_target = val_data[-1]
    test_context = test_context[None, :, :]
    test_target = test_target[None, :, :]

    sp.plot_mel_spectrogram(np.squeeze(test_context.numpy()), sample_rate)
    sp.plot_mel_spectrogram(np.squeeze(test_target.numpy()), sample_rate)


    test_output = cnn(test_context).detach()
    sp.plot_mel_spectrogram(np.squeeze(test_output.numpy()), sample_rate)

    plt.scatter(range(len(train_losses)), train_losses)
    plt.show()


if __name__ == '__main__':
    main()

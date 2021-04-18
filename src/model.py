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

#TODO: general kernal size, stride size 
class Model(nn.Module):
    '''
    N: Number of filters in autoencoder
    K: kernel size
    S: stride
    '''

    def __init__(self, K=3, S=2, INTENSITY_MAX=100):
        super(Model, self).__init__()

        self.K, self.S, self.inten_max = K, S, INTENSITY_MAX

        self.encoder = Encoder(K, S)
        self.decoder = Decoder(K, S)

        #mess around with reasonable sounding fuctions
        self.normalize = nn.ReLU()

    def forward(self, x):
        out = self.encoder.forward(x)
        out = self.decoder.forward(out)

        #remove normalization if needed
        out = self.normalize(x)

        return out

class Encoder(nn.Module):
    '''
    input: 128 x 128 x 1 vector
    output: 128 x 32 x 1 vector
    filter size 3
    stride 2
    [CONV-CONV-POOL2]×3 architecture, followed by 2 fully connected layers or 2 convolutional layers.

    '''
    def __init__(self, K=3, S=2):

        super(Encoder, self).__init__()
        self.K, self.S = K, S

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=K, stride=S, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=K, stride=S, padding=1),
            nn.ReLU()
            #nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )

        self.bn1 = nn.BatchNorm2d(32)

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=K, stride=S, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=K, stride=S, padding=1),
            nn.ReLU()
            #nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )

        self.bn2 = nn.BatchNorm2d(128)

        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=K, stride=S, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=K, stride=S, padding=1),
            nn.ReLU()
            #nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.bn3 = nn.BatchNorm2d(512)

        self.layer4 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=K, stride=S, padding=1),
            nn.ReLU()
        )
        self.bn4 = nn.BatchNorm2d(1024)

        self.fc1 = nn.Sequential(
            nn.Linear(1024, 1024), 
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024), 
            nn.ReLU()
        )


    def forward(self, x):
        out = self.layer1(x)
        out = self.bn1(out)
        out = self.layer2(out)
        out = self.bn2(out)
        out = self.layer3(out)
        out = self.bn3(out)
        out = self.layer4(out)
        out = self.bn4(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class Decoder(nn.Module):

    def __init__(self, K=3, S=2):
        
        super(Decoder, self).__init__()
        self.K, self.S = K, S

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=K, stride=S, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=K, stride=S, padding=1),
            nn.ReLU()
        )

        self.bn1 = nn.BatchNorm2d(256)

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=K, stride=S, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=K, stride=S, padding=1),
            nn.ReLU()
        )

        self.bn2 = nn.BatchNorm2d(64)

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=K, stride=S, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=K, stride=S, padding=1),
            nn.ReLU()
        )

        self.bn3 = nn.BatchNorm2d(16)

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(16, 1, kernel_size=K, stride=S, padding=1),
            nn.ReLU()
        )
        
        self.bn4 = nn.BatchNorm2d(1024)


    def forward(self, x):
        out = self.layer1(x)
        out = self.bn1(out)
        out = self.layer2(out)
        out = self.bn2(out)
        out = self.layer3(out)
        out = self.bn3(out)
        out = self.layer4(out)
        out = self.bn4(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

def main():
    m = Model()

if __name__ == "__main__":
    main()

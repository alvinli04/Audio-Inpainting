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

class Model(torch.nn.Module):
	'''
	N: Number of filters in autoencoder
    L: Length of the filters (in samples)
    K: kernel size
    S: stride
    '''

#TODO: general kernal size, stride size 
class Encoder(torch.nn.Module):
	'''
	input: 128 x 128 x 1 vector
	output: 128 x 32 x 1 vector
	filter size 3
	stride 2
	[CONV-CONV-POOL2]×3 architecture, followed by 2 fully connected layers or 2 convolutional layers.

	'''
	def __init__(self, L, N, K=3, S=2):

		super(Encoder, self).__init__()
		self.L, self.N, self.K, self.S = L, N, K, S

		self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=K, stride=S),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=K, stride=S),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

		self.bn1 = nn.BatchNorm2d(32)
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=K, stride=S),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=K, stride=S),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.bn2 = nn.BatchNorm2d(128)
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=K, stride=S),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=K, stride=S),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.bn3 = nn.BatchNorm2d(512)

	def forward(self, x):
		out = self.layer1(x)
        out = self.layer2(out)
        return out

class Decoder(torch.nn.Module):

	def __init__(self, L, N):


	def forward(self, mixture):
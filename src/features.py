# -*- coding: utf-8 -*-
"""
Definition of functions for extracting spectrogram, MFCC-gram, and scalogram
from waveform data.

Created on Tue Oct 31 18:15:17 2023

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""


import os
import numpy as np
import librosa
from ssqueezepy import cwt, stft



# Load a wave data
def load_wave_data(audio_dir, file_name):
    file_path = os.path.join(audio_dir, file_name)
    x, fs = librosa.load(file_path, sr=None)

    return x, fs



# Extract the STFT spectrogram
def extract_spectrogram(x, hop_length=1, n_fft=2048):
    S = np.abs(librosa.stft(x, hop_length=hop_length, n_fft=n_fft))
    # print("Shape of spectrogram:", S.shape)

    return S



# Extract the STFT MFCC-gram
def extract_MFCCgram(S, sr=22050, n_mfcc=128):
    # M = librosa.feature.mfcc(y=x, sr=sr)
    SM = librosa.feature.melspectrogram(S=S**2, sr=sr, n_mels=n_mfcc)
    M = librosa.feature.mfcc(S=librosa.power_to_db(SM, ref=np.max), sr=sr, n_mfcc=n_mfcc)
    # print("Shape of MFCC-gram:", M.shape)

    return M



# Extract the STFT scalogram
def extract_scalogram(x, fs=22050, wavelet='morlet'):
    W, scales = cwt(x, wavelet=wavelet, fs=fs)
    W = np.abs(W)
    # print("Shape of scalogram:", W.shape)

    return W



def scale(matrix):
    # Perform min-max scaling
    min_val = np.min(matrix)
    max_val = np.max(matrix)

    scaled_matrix = (matrix - min_val) / (max_val - min_val)

    return scaled_matrix

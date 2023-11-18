# -*- coding: utf-8 -*-
"""
Script for loading the dataset related to Construction Sites and
extracting scalogram after resampling audio to 22,050 Hz.

Created on Sun Nov  5 13:15:52 2023

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""


import os
import numpy as np
import librosa
import cv2
import features as ft
import yaml


# Loading the configuration file
config_file = 'CS.yaml'
with open(config_file, 'r') as f:
    config = yaml.load(f, Loader=yaml.Loader)

# print(config)


# Set data folder
data_folder = config['data_folder']
save_folder = config['save_folder']

sets = ['Training/', 'Testing/']


# Set resample rate and image size
f_res = config['f_res']   # resample rate
N_res = config['N_res']   # resize dimension


# Main loop
for s in sets:
    directory = data_folder + s
    audio_directories = os.listdir(directory)
    audio_directories.sort()

    c = 0  # Class label
    feat_W = []
    labels = []

    for d in audio_directories:
        path_directories = directory + d
        file_list = os.listdir(path_directories)
        N = len(file_list)
        n = 0

        for f in file_list:
            file_name = path_directories + '/' + f

            # Load wave file
            x, fs = librosa.load(file_name, sr=None)
            x = librosa.resample(x, orig_sr=fs, target_sr=f_res)

            # Extract features
            W = ft.extract_scalogram(x, wavelet='morlet')

            # Resize features to 224x224 or 227x227
            W = cv2.resize(W, dsize=(N_res, N_res), interpolation=cv2.INTER_LINEAR)

            # Normalize features
            W = ft.scale(W)

            # Transform as an integer image
            W = np.array(255*W, dtype='uint8')

            # Set label
            y = c

            # Append features and labels
            feat_W.append(W)
            labels.append(y)

            # Print advancement
            n += 1
            if (n % 100):
                print("\r{}: {}%".format(d, round(100*n/N, 1)), end='')


        c += 1
        print("\r{}: {}%".format(d, 100.0), end='\n')


    # Saving data
    save_path = save_folder + s

    np.save(save_path + 'scalograms.npy', feat_W)
    np.save(save_path + 'labels.npy', labels)

    print("Done for folder: ", s)


print("Done!")

# -*- coding: utf-8 -*-
"""
Script for the classification of audio signals recorded in construction sites
by using scalogram input.

Created on Mon Nov  6 16:07:52 2023

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""

import numpy as np
import models
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import yaml


# Loading the configuration file
config_file = 'CS.yaml'
with open(config_file, 'r') as f:
    config = yaml.load(f, Loader=yaml.Loader)

# print(config)


# Set main hyper-parameters
LR  = config['LR']    # Learning rate
N_b = config['N_b']   # Batch size
N_e = config['N_e']   # Number of epochs


# Set data folder
data_folder = config['data_folder']
save_folder = config['save_folder']
result_folder = config['result_folder']

sets = ['Training/', 'Testing/']


# %% Load training set
training_folder = data_folder + sets[0]

X = np.load(training_folder + 'scalograms.npy')
y = np.load(training_folder + 'labels.npy')


# %% Shuffle the dataset
np.random.seed(seed=42)
idx = np.random.permutation(len(y))
X = X[idx,:,:]
y = y[idx]


# Convert in One-Hot encoding
y_cat = to_categorical(y, 5)


# %% Select the model
net = models.AlexNet1(LR)
model_name = config['model_name']

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=2)


# %% Train the selected model
history = net.fit(X, y_cat, batch_size=N_b, epochs=N_e, validation_split=0.1, shuffle=True, callbacks=[early_stop])


# Save the trained model
save_file = save_folder + model_name + '.h5'
net.save(save_file, overwrite=True, include_optimizer=True, save_format='h5')

np.save(save_folder + model_name + '_history.npy', history.history)
# history = np.load(save_folder + model_name + '_history.npy', allow_pickle='TRUE').item()


# %% Plot loss curve
ep = range(1, N_e+1)
plt.figure()
plt.plot(ep, history.history['loss'], linewidth=2, label='Training loss')
plt.plot(ep, history.history['val_loss'], linewidth=2, label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
# plt.show()
# fig_name = save_folder + model_name + 'Training_Loss.pdf'
# plt.savefig(fig_name, format='pdf')


# Plot accuracy curve
ep = range(1, N_e+1)
plt.figure()
plt.plot(ep, history.history['accuracy'], linewidth=2, label='Training accuracy')
plt.plot(ep, history.history['val_accuracy'], linewidth=2, label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
# plt.show()
# fig_name = save_folder + model_name + 'Training_Accuracy.pdf'
# plt.savefig(fig_name, format='pdf')




# %% Testing

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report


del X, y

# Load test set
test_folder = data_folder + sets[1]

Xt = np.load(test_folder + 'scalograms.npy')
yt = np.load(test_folder + 'labels.npy')


# Convert in One-Hot encoding
yt_cat = to_categorical(yt, 5)


# Load the trained model
# net = tensorflow.keras.models.load_model(save_file)


# %% Evaluate the model
results = net.evaluate(Xt, yt_cat)
print('Final Loss:', results[0])
print('Overall Accuracy:', results[1])


# Evaluate the model output for test set
y_pred = net.predict(Xt)
y_pred = np.argmax(y_pred, axis =1)


# Evaluating the trained model
acc = accuracy_score(yt, y_pred)
pre = precision_score(yt, y_pred, average='weighted')
rec = recall_score(yt, y_pred, average='weighted')
f1  = f1_score(yt, y_pred, average='weighted')


# Printing metrics
print("Overall accuracy: {}%".format(round(100*acc,2)))
print("Precision: {}".format(round(pre,3)))
print("Recall: {}".format(round(rec,3)))
print("F1-score: {}".format(round(f1,3)))
print(" ", end='\n')
print("Complete report: ", end='\n')
print(classification_report(yt, y_pred))
print(" ", end='\n')


# Showing CM results
labels = ['JD505D', 'IRCOM', 'Mixer', 'CAT320E', 'Hitachi50U']

cm = confusion_matrix(yt, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = labels)
disp.plot(cmap='Blues')


# Save results on a text file
res_file = result_folder + 'Results_' + model_name + '.txt'
with open(res_file, 'a') as results:  # save the results in a .txt file
      results.write('-------------------------------------------------------\n')
      results.write('Acc: %s\n' % round(100*acc,2))
      results.write('Pre: %s\n' % round(pre,3))
      results.write('Rec: %s\n' % round(rec,3))
      results.write('F1: %s\n\n' % round(f1,3))
      results.write(classification_report(yt, y_pred, digits=3))
      results.write('\n\n')

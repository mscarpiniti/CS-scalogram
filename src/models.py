# -*- coding: utf-8 -*-
"""
This file defines the models to be used for the classification of audio
recorded in construction sites by using multiview early data fusion.

Created on Mon Nov  6 15:47:19 2023

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""

# TensorFlow ≥2.0 is required
from tensorflow import keras



# Create the AlexNet with 1 channel
def AlexNet1(LR=0.001):

    alex = keras.models.Sequential([
        keras.layers.Rescaling(scale=1./255, input_shape=(227,227,1)),
        keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        #keras.layers.Dense(1000, activation='softmax')
        keras.layers.Dense(5, activation='softmax')
    ])

    # Display the model's architecture
    # alex.summary()

    alex.compile(loss='categorical_crossentropy',
                  # optimizer='nadam',
                  # optimizer=tf.optimizers.Nadam(learning_rate=LR),
                  optimizer=keras.optimizers.Nadam(learning_rate=LR),
                  metrics=['accuracy'])

    return alex

"""CNN for walking detection"""

# Authors: Julia Camps <julia.camps.sereix@est.fib.upc.edu>
# Created on: 06/10/2016 17:22

import numpy as np
import random as rd

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.callbacks import Callback
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D


def build_small_CNN(time_step=100, n_feature=9, n_chan=1):
    """Build the model"""
    model_ = Sequential()
    
    model_.add(Convolution2D(64, 9, 9, border_mode='same',
                             input_shape=(time_step, n_feature,
                                          n_chan), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model_.add(Dropout(0.25))
    
    model_.add(Convolution2D(64, 5, 9, activation='relu'))
    # model_.add(MaxPooling2D(pool_size=(2, 1)))
    model_.add(Dropout(0.25))
    
    model_.add(Convolution2D(64, 5, 1, activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model_.add(Dropout(0.25))
    
    model_.add(Convolution2D(64, 3, 1, activation='relu'))
    # # model.add(MaxPooling2D(pool_size=(2, 2)))
    model_.add(Dropout(0.25))
    
    model_.add(Flatten())
    model_.add(Dense(128, activation='relu'))
    model_.add(Dropout(0.25))
    
    model_.add(Dense(128, activation='relu'))
    model_.add(Dropout(0.5))
    
    model_.add(Dense(1, activation='sigmoid'))
    # model_.add(Dense(_N_CLASS, activation='softmax'))
    
    return model_
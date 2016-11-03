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
from keras.layers.core import SpatialDropout2D


def build_model(window_size, n_feature=9, n_chan=1, n_conv=1,
                n_dense=1, k_shapes=[[64, 9, 9]], dense_shape=[128],
                opt_name='adadelta', pooling=False, dropout=0.5):
    """Build the model"""
    model_ = Sequential()
    name = ''
    for i in range(n_conv):
        n_kernel = k_shapes[i][0]
        h_kernel = k_shapes[i][1]
        w_kernel = k_shapes[i][2]
        model_.add(Convolution2D(n_kernel, h_kernel, w_kernel,
                                 border_mode='same',
                                 input_shape=(window_size,
                                              n_feature, n_chan),
                                 activation='relu'))
        name = (name + 'C(' + str(n_kernel) + ', '
                + str(h_kernel) + str(w_kernel)) + ')-'
        if pooling:
            model_.add(MaxPooling2D(pool_size=(2, 1)))
            name = name + 'P-'
        if dropout > 0:
            model_.add(SpatialDropout2D(dropout))
            name = name + 'DR(' + str(dropout) + ')-'
    model_.add(Flatten())
    for i in range(n_dense):
        model_.add(Dense(dense_shape[i], activation='relu'))
        name = name + 'DN(' + str(dense_shape[i]) + ')-'
        
        if dropout > 0:
            model_.add(Dropout(dropout))
            name = name + 'D(' + str(dropout) + ')-'

    model_.add(Dense(1, activation='sigmoid'))
    name = name + 'DN(1, Sigmoid)|'
    
    model_.compile(loss='binary_crossentropy',
                   optimizer=opt_name,
                   metrics=['accuracy'])
    name = name + opt_name
        
    return [name, model_]


# EOF

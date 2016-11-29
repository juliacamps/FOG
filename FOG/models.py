"""CNN for walking detection"""

# Authors: Julia Camps <julia.camps.sereix@est.fib.upc.edu>
# Created on: 06/10/2016 17:22

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import AtrousConvolution1D
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.regularizers import l1, l2
import keras.backend as K


def mean_pred(y_true, y_pred):
    """"""
    return K.mean(y_pred)


def build_model(window_size, n_feature=9, n_conv=1, n_dense=1,
                k_shapes=[[32, 3]], dense_shape=[128],
                init='uniform', opt_name='adadelta',
                pooling=False, dropout=0.5, atrous=False,
                regularizer=None):
    """Build the model"""
    if regularizer is None:
        regular = None
    else:
        if regularizer == 'l1':
            regular = l1(l=0.01)
        elif regularizer == 'l2':
            regular = l2(l=0.01)
        else:
            # Not accepted option
            print('ERROR: Regularizer is Undefined')
            regular = None
            
    model_ = Sequential()
    name = ''
    if n_conv > 0:
        nb_kernel = k_shapes[0][0]
        he_kernel = k_shapes[0][1]
        if atrous:
            model_.add(Convolution1D(nb_kernel, he_kernel,
                                 init=init,
                                 W_regularizer=regular,
                                 border_mode='same',
                                 input_shape=(window_size, n_feature),
                                 activation='relu'))
            name += ('C(' + str(nb_kernel) + ',' + str(he_kernel)
                     + ')-')
        else:
            model_.add(AtrousConvolution1D(nb_kernel, he_kernel,
                                       atrous_rate=2,
                                       init=init,
                                       W_regularizer=regular,
                                       border_mode='same',
                                       input_shape=(window_size,
                                                    n_feature),
                                       activation='relu'))
            name += ('A(' + str(nb_kernel) + ',' + str(he_kernel)
                     + ')-')
        if pooling:
            model_.add(MaxPooling1D(pool_length=2))
            name += 'P-'
        if dropout > 0:
            model_.add(Dropout(dropout))
            name += 'DR(' + str(dropout) + ')-'
    for i in range(1, n_conv):
        nb_kernel = k_shapes[i][0]
        he_kernel = k_shapes[i][1]
        model_.add(Convolution1D(nb_kernel, he_kernel, init=init,
                                 W_regularizer=regular,
                                 border_mode='same',
                                 activation='relu'))
        name += 'C(' + str(nb_kernel) + ',' + str(he_kernel) + ')-'
        if pooling:
            model_.add(MaxPooling1D(pool_length=2))
            name += 'P-'
        if dropout > 0:
            model_.add(Dropout(dropout))
            name += 'DR(' + str(dropout) + ')-'
    model_.add(Flatten())
    for i in range(n_dense):
        model_.add(Dense(dense_shape[i], activation='relu',
                         init=init, W_regularizer=regular))
        name += 'DN(' + str(dense_shape[i]) + ')-'
        
        if dropout > 0:
            model_.add(Dropout(dropout))
            name += 'D(' + str(dropout) + ')-'

    model_.add(Dense(1, activation='sigmoid',
                     W_regularizer=regular))
    name += 'DN(1,Sigmoid)|INIT:' + init + '|REGULAR:' + str(regular)
    
    model_.compile(loss='binary_crossentropy',
                   optimizer=opt_name,
                   metrics=[])  # 'accuracy', 'precision',
                                # 'recall', mean_pred])
    name += '|OPT:' + opt_name
    
    return [name, model_]


# EOF

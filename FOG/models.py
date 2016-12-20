"""CNN for walking detection"""

# Authors: Julia Camps <julia.camps.sereix@est.fib.upc.edu>
# Created on: 06/10/2016 17:22

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import AtrousConvolution1D
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.regularizers import l1, l2
from keras.optimizers import Adam

from FOG.experiment_conf import get_n_feature


def build_model(window_size, n_conv, n_dense, k_shapes,
                dense_shape, init, opt_name, pooling, dropout,
                atrous, regularizer, temporal, learning_rate):
    """Build the model"""
    n_feature = get_n_feature()
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
            
    model = Sequential()
    model_structure = ''
    if n_conv > 0:
        nb_kernel = k_shapes[0][0]
        he_kernel = k_shapes[0][1]
        if atrous:
            model.add(AtrousConvolution1D(nb_kernel, he_kernel,
                                          atrous_rate=2,
                                          init=init,
                                          W_regularizer=regular,
                                          border_mode='same',
                                          input_shape=(window_size,
                                                       n_feature),
                                          activation='relu'))
            model_structure += ('A(' + str(nb_kernel) + ','
                                + str(he_kernel) + ')-')
        else:
            model.add(Convolution1D(nb_kernel, he_kernel,
                                    init=init,
                                    W_regularizer=regular,
                                    border_mode='same',
                                    input_shape=(window_size,
                                                 n_feature),
                                    activation='relu'))
            model_structure += ('C(' + str(nb_kernel) + ','
                                + str(he_kernel) + ')-')
            
        if pooling:
            model.add(MaxPooling1D(pool_length=2))
            model_structure += 'P-'
        if dropout > 0:
            model.add(Dropout(dropout))
            model_structure += 'DR(' + str(dropout) + ')-'
    for i in range(1, n_conv):
        nb_kernel = k_shapes[i][0]
        he_kernel = k_shapes[i][1]
        model.add(Convolution1D(nb_kernel, he_kernel, init=init,
                                 W_regularizer=regular,
                                 border_mode='same',
                                 activation='relu'))
        model_structure += ('C(' + str(nb_kernel) + ',' + str(
            he_kernel) + ')-')
        if pooling:
            model.add(MaxPooling1D(pool_length=2))
            model_structure += 'P-'
        if dropout > 0:
            model.add(Dropout(dropout))
            model_structure += 'DR(' + str(dropout) + ')-'
    
    if temporal:
        for i in range(n_dense):
            model.add(LSTM(dense_shape[i], activation='relu',
                           init=init))
            model_structure += 'LSTM(' + str(dense_shape[i]) + ')-'
        
            # if dropout > 0:
            #     model.add(Dropout(dropout))
            #     model_structure += 'D(' + str(dropout) + ')-'
    else:
        model.add(Flatten())
        for i in range(n_dense):
            model.add(Dense(dense_shape[i], activation='relu',
                             init=init, W_regularizer=regular))
            model_structure += 'DN(' + str(dense_shape[i]) + ')-'
            
            if dropout > 0:
                model.add(Dropout(dropout))
                model_structure += 'D(' + str(dropout) + ')-'
    
    model.add(Dense(1, activation='sigmoid',
                     W_regularizer=regular))
    model_structure += ('DN(1,Sigmoid)|INIT:' + init + '|REGULAR:'
                        + str(regular))
    if opt_name == 'adam':
        opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999,
                    epsilon=1e-08, decay=0.0)
    else:
        opt = None
    model.compile(loss='binary_crossentropy',
                  optimizer=opt, metrics=[])
    model_structure += ('|OPT:' + opt_name + '(lr=' + str(
        learning_rate)
                        + ')')
    
    return [model_structure, model]


# EOF

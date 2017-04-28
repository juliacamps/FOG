"""CNN for walking detection"""

# Authors: Julia Camps <julia.camps.sereix@est.fib.upc.edu>
# Created on: 06/10/2016 17:22

from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import GRU
# from keras.layers import concatenate
from keras.layers import Dropout
from keras.layers import Flatten
# from keras.layers.convolutional import AtrousConvolution1D
from keras.layers.convolutional import Convolution1D
# from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.regularizers import l1, l2
from keras.optimizers import Adam
from keras import backend as K
import keras

from keras.engine import merge


_PENALTY_WEIGHT = 1.


def hinge_loss(y_true, y_pred):
    """"""
    return K.mean(K.maximum(1. - y_true * y_pred, 0.), axis=-1)


def weighted_hinge_loss(y_true, y_pred):
    """"""
    y_pos = K.maximum(y_true, 0.)
    y_neg = K.minimum(y_true, 0.)
    hinge_pos = K.mean(K.maximum(1. - y_pos * y_pred, 0.)) * (
        1.-_PENALTY_WEIGHT)
    hinge_neg = K.mean(K.maximum(1. - y_neg * y_pred, 0.)) * \
                _PENALTY_WEIGHT
    return hinge_pos + hinge_neg


def compile_model(model, objective, penalty, learning_rate, optimizer_name):
        """"""
        loss = None
        optimizer = None
        if optimizer_name == 'adam':
            optimizer = Adam(lr=learning_rate, beta_1=0.9,
                             beta_2=0.999, epsilon=1e-08, decay=0.0, clipvalue=1.)
        if objective == 'w_hinge':
            global _PENALTY_WEIGHT
            _PENALTY_WEIGHT = penalty
            loss = weighted_hinge_loss
        elif objective == 'hinge':
            # loss = hinge_loss
            loss = 'hinge'
            
        model.compile(loss=loss, optimizer=optimizer, metrics=None)


def build_model(spectral_input_size, temporal_input_size,
                spectral_n_feature, temporal_n_feature,
                conv_layers, dense_layers, init,
                learning_rate, optimizer, pooling, dropout, atrous,
                regularizer_conf, temporal, objective, penalty,
                activation, last_layer, batch_size,
                n_batch_per_file, lstm_dropout):
    """Build the model"""

    spectral_input = Input(
        batch_shape=(batch_size, spectral_input_size,
                     spectral_n_feature),
        name='spectral_input')
    temporal_input = Input(
        batch_shape=(batch_size, temporal_input_size,
                     temporal_n_feature),
        name='temporal_input')
    pre_temporal_input = Input(
        batch_shape=(batch_size, temporal_input_size,
                     temporal_n_feature),
        name='pre_temporal_input')
    n_conv = len(conv_layers)
    n_dense = len(dense_layers)

    nb_kernel = conv_layers[0][0]
    he_kernel = conv_layers[0][1]
    regularizer = None
    if regularizer_conf['name'] == 'l1':
        regularizer = l1(l=regularizer_conf['value'])
    elif regularizer_conf['name'] == 'l2':
        regularizer = l2(l=regularizer_conf['value'])

    # spectral_conv = Conv1D(nb_kernel, he_kernel, strides=1, padding='valid',
    #                        dilation_rate=1, activation=activation,
    #                        use_bias=True,
    #                        kernel_initializer=init,
    #                        bias_initializer='zeros',
    #                        kernel_regularizer=regularizer,
    #                        bias_regularizer=None,
    #                        activity_regularizer=None,
    #                        kernel_constraint=None,
    #                        bias_constraint=None)(spectral_input)
    spectral_conv = Convolution1D(nb_kernel, he_kernel,
                                  border_mode='valid',
                           activation=activation,
                           bias=True,
                                  init=init,
                                  W_regularizer=regularizer)(spectral_input)
    # spectral_conv = MaxPooling1D(pool_length=2)(spectral_conv)
    spectral_conv = Dropout(dropout)(spectral_conv)
    # spectral_conv = MaxPooling1D(pool_size=2, strides=2,
    #                              padding='valid')(spectral_conv)
    # temporal_conv = Conv1D(nb_kernel, he_kernel, strides=1, padding='valid',
    #                        dilation_rate=1, activation=activation,
    #                        use_bias=True,
    #                        kernel_initializer=init,
    #                        bias_initializer='zeros',
    #                        kernel_regularizer=regularizer,
    #                        bias_regularizer=None,
    #                        activity_regularizer=None,
    #                        kernel_constraint=None,
    #                        bias_constraint=None)(temporal_input)
    temporal_conv = Convolution1D(nb_kernel, he_kernel,
                                  border_mode='valid',
                           activation=activation,
                           bias=True,
                                  init=init,
                                  W_regularizer=regularizer)(temporal_input)
    
    # temporal_conv = MaxPooling1D(pool_size=2, strides=2,
    #                              padding='valid')(temporal_conv)
    # temporal_conv = MaxPooling1D(pool_length=2)(temporal_conv)
    temporal_conv = Dropout(dropout)(temporal_conv)
    pre_temporal_conv = Convolution1D(nb_kernel, he_kernel,
                                  border_mode='valid',
                                  activation=activation,
                                  bias=True,
                                  init=init,
                                  W_regularizer=regularizer)(pre_temporal_input)
    # pre_temporal_conv = MaxPooling1D(pool_length=2)(pre_temporal_conv)
    pre_temporal_conv = Dropout(dropout)(pre_temporal_conv)
    for i in range(1, n_conv):
        nb_kernel = conv_layers[i][0]
        he_kernel = conv_layers[i][1]
        regularizer = None
        if regularizer_conf['name'] == 'l1':
            regularizer = l1(l=regularizer_conf['value'])
        elif regularizer_conf['name'] == 'l2':
            regularizer = l2(l=regularizer_conf['value'])
        # spectral_conv = Conv1D(nb_kernel, he_kernel, strides=1, padding='valid',
        #                    dilation_rate=1, activation=activation,
        #                    use_bias=True,
        #                    kernel_initializer=init,
        #                    bias_initializer='zeros',
        #                    kernel_regularizer=None,
        #                    bias_regularizer=None,
        #                    activity_regularizer=regularizer,
        #                    kernel_constraint=None,
        #                    bias_constraint=None)(spectral_conv)
        spectral_conv = Convolution1D(nb_kernel, he_kernel,
                                  border_mode='valid',
                           activation=activation,
                           bias=True,
                                  init=init,
                                  W_regularizer=None)(
            spectral_conv)
        spectral_conv = Dropout(dropout)(spectral_conv)
        # temporal_conv = Conv1D(nb_kernel, he_kernel, strides=1, padding='valid',
        #                    dilation_rate=1, activation=activation,
        #                    use_bias=True,
        #                    kernel_initializer=init,
        #                    bias_initializer='zeros',
        #                    kernel_regularizer=None,
        #                    bias_regularizer=None,
        #                    activity_regularizer=regularizer,
        #                    kernel_constraint=None,
        #                    bias_constraint=None)(temporal_conv)
        temporal_conv = Convolution1D(nb_kernel, he_kernel,
                                  border_mode='valid',
                           activation=activation,
                           bias=True,
                                  init=init,
                                  W_regularizer=None)(
            temporal_conv)
        # temporal_conv = MaxPooling1D(pool_size=2, strides=2,
        #                              padding='valid')(temporal_conv)
        # if i < 3:
        #     temporal_conv = MaxPooling1D(pool_length=2)(temporal_conv)
        temporal_conv = Dropout(dropout)(temporal_conv)
        
        pre_temporal_conv = Convolution1D(nb_kernel, he_kernel,
                                      border_mode='valid',
                                      activation=activation,
                                      bias=True,
                                      init=init,
                                      W_regularizer=None)(
            pre_temporal_conv)
        # if i < 3:
        #     pre_temporal_conv = MaxPooling1D(pool_length=2)(pre_temporal_conv)
        pre_temporal_conv = Dropout(dropout)(pre_temporal_conv)
    
    # spectral_conv = GRU(dense_layers[0], activation='tanh',
    #                     recurrent_activation='hard_sigmoid',
    #                     use_bias=True,
    #                     kernel_initializer='glorot_uniform',
    #                     recurrent_initializer='orthogonal',
    #                     bias_initializer='zeros',
    #                     kernel_regularizer=None,
    #                     recurrent_regularizer=None,
    #                     bias_regularizer=None,
    #                     activity_regularizer=None,
    #                     kernel_constraint=None,
    #                     recurrent_constraint=None,
    #                     bias_constraint=None,
    #                     dropout=0.0,
    #                     stateful=True,
    #                     implementation=0,
    #                     recurrent_dropout=lstm_dropout)(spectral_conv)
    # temporal_conv = GRU(dense_layers[0], activation='tanh',
    #                     recurrent_activation='hard_sigmoid',
    #                     use_bias=True,
    #                     kernel_initializer='glorot_uniform',
    #                     recurrent_initializer='orthogonal',
    #                     bias_initializer='zeros',
    #                     kernel_regularizer=None,
    #                     recurrent_regularizer=None,
    #                     bias_regularizer=None,
    #                     activity_regularizer=None,
    #                     kernel_constraint=None,
    #                     recurrent_constraint=None,
    #                     bias_constraint=None,
    #                     dropout=0.0,
    #                     stateful=True,
    #                     implementation=0,
    #                     recurrent_dropout=lstm_dropout)(temporal_conv)
    spectral_conv = Flatten()(spectral_conv)
    temporal_conv = Flatten()(temporal_conv)
    pre_temporal_conv = Flatten()(pre_temporal_conv)
    # merged_conv = concatenate([spectral_conv, temporal_conv])
    merged_conv = merge([spectral_conv, temporal_conv, pre_temporal_conv],
                        mode='concat')

    for i in range(0, n_dense):
        regularizer = None
        if regularizer_conf['name'] == 'l1':
            regularizer = l1(l=regularizer_conf['value'])
        elif regularizer_conf['name'] == 'l2':
            regularizer = l2(l=regularizer_conf['value'])
        
        # merged_conv = Dense(dense_layers[i],
        #                     activation=activation, kernel_initializer=init,
        #                     activity_regularizer=None, use_bias=True,
        #                     bias_initializer='zeros',
        #                     kernel_regularizer=regularizer,
        #                     bias_regularizer=None,
        #                     kernel_constraint=None, bias_constraint=None
        #                     )(merged_conv)
        merged_conv = Dense(dense_layers[i],
                            activation=activation,
                            init=init, W_regularizer=l2(l=1e-02))(
            merged_conv)
        merged_conv = Dropout(lstm_dropout)(merged_conv)
        
    last_regularizer = None
    if last_layer['regularization']['name'] == 'l1':
        last_regularizer = l1(l=last_layer['regularization']['value'])
    elif last_layer['regularization']['name'] == 'l2':
        last_regularizer = l2(l=last_layer['regularization']['value'])
    # output = Dense(1, activation=last_layer['activation'],
    #                kernel_initializer=init,
    #                activity_regularizer=None, use_bias=True,
    #                bias_initializer='zeros',
    #                kernel_regularizer=last_regularizer,
    #                bias_regularizer=None,
    #                kernel_constraint=None, bias_constraint=None,
    #                name='output')(merged_conv)
    output = Dense(1, activation=last_layer['activation'],
                   init=init,
                   W_regularizer=last_regularizer,name='output')(merged_conv)

    # model = Model(inputs=[spectral_input, temporal_input],
    #               outputs=[output])
    model = Model(input=[spectral_input, temporal_input,
                         pre_temporal_input],
                  output=output)
    
    
    compile_model(model, objective, penalty, learning_rate, optimizer)


    model_structure = ''
    print(model.summary())

    return [model_structure, model]


# EOF

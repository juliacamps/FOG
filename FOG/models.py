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
from keras.regularizers import l1, l2, l1l2, activity_l1, \
    activity_l2, activity_l1l2
from keras.optimizers import Adam
from keras import backend as K
# from keras.backend import floatx

from FOG.experiment_conf import get_n_feature

# _RESULTS = K.zeros((4,), dtype=K.floatx())

#
# t1 = K.floatx()
#
# def set_val(e):
#     global t1
#     t1 = e
# sess = K.get_session()
# # sess.run(set_val(3))
# print(sess.run(_RESULTS[0]))

_PENALTY_WEIGHT = 1.

def specificity(y_true, y_pred):
    '''Calculates the specificity, a metric for multi-label classification of
    how many relevant items are selected.
    '''
    negatives = K.maximum(K.clip(-y_true, -1., 1.), 0.)
    true_negatives = K.sum(K.maximum(negatives * K.sign(K.clip(
        -y_pred, -1., 1.)), 0.))
    specificity = true_negatives / (K.sum(negatives) + 1e-20)
    return specificity


def partial_metrics(y_true, y_pred):
    '''Calculates the specificity, a metric for multi-label classification of
    how many relevant items are selected.
    '''
    positives = K.maximum(K.clip(y_true, -1., 1.), 0.)
    negatives = 1. - positives
    pred_positives = K.maximum(K.sign(K.clip(y_pred, -1., 1.)), 0.)
    pred_negatives = 1. - pred_positives
    true_positives = K.sum(positives * pred_positives)
    true_negatives = K.sum(negatives * pred_negatives)
    return {'TN': true_negatives, 'N': K.sum(negatives),
            'TP': true_positives, 'P': K.sum(positives)}


def sensitivity(y_true, y_pred):
    '''Calculates the sensitivity.'''
    positives = K.maximum(K.clip(y_true, -1., 1.), 0.)
    true_positives = K.sum(K.maximum(positives * K.sign(K.clip(y_pred,
                                                               -1., 1.)), 0.))
    sensitivity = true_positives / (K.sum(positives) + 1e-20)
    return sensitivity


def acc(y_true, y_pred):
    """"""
    return K.mean(K.maximum(y_true * K.sign(y_pred), 0.))


# def results(y_true, y_pred):
#     """"""
#     global _RESULTS
#     negatives = K.maximum(-y_true, 0.)
#     true_negatives = K.sum(K.maximum(negatives * K.sign(-y_pred), 0.))
#     positives = K.maximum(y_true, 0.)
#     true_positives = K.sum(K.maximum(positives * K.sign(y_pred), 0.))
#     _RESULTS[0] += positives
#     _RESULTS[1] += true_positives
#     _RESULTS[2] += negatives
#     _RESULTS[3] += true_negatives
#     return {
#             'P': K.sum(positives),
#             'N': K.sum(negatives), 'TP': true_positives, 'TN':
#                 true_negatives
#             }


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


def compile_model(model, objective, optimization):
        """"""
        loss = None
        optimizer = None
        if optimization['name'] == 'adam':
            optimizer = Adam(lr=optimization['lr'], beta_1=0.9,
                             beta_2=0.999, epsilon=1e-08, decay=0.0)
        if objective['name'] == 'w_hinge':
            global _PENALTY_WEIGHT
            _PENALTY_WEIGHT = objective['penalty']
            loss = weighted_hinge_loss
        elif objective['name'] == 'hinge':
            loss = hinge_loss
            
        model.compile(loss=loss, optimizer=optimizer, metrics=None)


def build_model(window_size, conv_layers, dense_layers, init,
                optimization, pooling_layers, dropout, atrous,
                regularizer_conf, temporal, objective,
                activation, last_layer):
    """Build the model"""
    n_feature = get_n_feature()
            
    model = Sequential()
    model_structure = ''
    n_conv = len(conv_layers)
    n_dense = len(dense_layers)
    if n_conv > 0:
        nb_kernel = conv_layers[0][0]
        he_kernel = conv_layers[0][1]
        regularizer = None
        if regularizer_conf['name'] == 'l1':
            regularizer = activity_l1(l=regularizer_conf['value'])
        elif regularizer_conf['name'] == 'l2':
            regularizer = activity_l2(l=regularizer_conf['value'])
        elif regularizer_conf['name'] == 'l1l2':
            regularizer = activity_l1l2(l1=regularizer_conf['value'],
                               l2=regularizer_conf['value'])
        if atrous:
            model.add(AtrousConvolution1D(nb_kernel, he_kernel,
                                          atrous_rate=2,
                                          init=init,
                                          activity_regularizer=regularizer,
                                          border_mode='same',
                                          input_shape=(window_size,
                                                       n_feature),
                                          activation=activation))
            model_structure += ('A(' + str(nb_kernel) + ','
                                + str(he_kernel) + ')')
        else:
            model.add(Convolution1D(nb_kernel, he_kernel,
                                    init=init,
                                    activity_regularizer=regularizer,
                                    border_mode='same',
                                    input_shape=(window_size,
                                                 n_feature),
                                    activation=activation))
            model_structure += ('C(' + str(nb_kernel) + ','
                                + str(he_kernel) + ')')
            
        if pooling_layers[0]:
            model.add(MaxPooling1D(pool_length=2))
            model_structure += '-P'
        if dropout > 0:
            model.add(Dropout(dropout))
            model_structure += '-DR(' + str(dropout) + ')'
    for i in range(1, n_conv):
        nb_kernel = conv_layers[i][0]
        he_kernel = conv_layers[i][1]
        regularizer = None
        if regularizer_conf['name'] == 'l1':
            regularizer = activity_l1(l=regularizer_conf['value'])
        elif regularizer_conf['name'] == 'l2':
            regularizer = activity_l2(l=regularizer_conf['value'])
        elif regularizer_conf['name'] == 'l1l2':
            regularizer = activity_l1l2(l1=regularizer_conf['value'],
                               l2=regularizer_conf['value'])
        model.add(Convolution1D(nb_kernel, he_kernel, init=init,
                                activity_regularizer=regularizer,
                                 border_mode='same',
                                 activation=activation))
        model_structure += ('-C(' + str(nb_kernel) + ',' + str(
            he_kernel) + ')')
        if pooling_layers[i]:
            model.add(MaxPooling1D(pool_length=2))
            model_structure += '-P'
        if dropout > 0:
            model.add(Dropout(dropout))
            model_structure += '-DR(' + str(dropout) + ')'
    
    if temporal:
        for i in range(n_dense):
            model.add(LSTM(dense_layers[i], activation=activation,
                           init=init))
            model_structure += '-LSTM(' + str(dense_layers[i]) + ')'
        
            # if dropout > 0:
            #     model.add(Dropout(dropout))
            #     model_structure += 'D(' + str(dropout) + ')-'
    else:
        model.add(Flatten())
        for i in range(n_dense):
            regularizer = None
            if regularizer_conf['name'] == 'l1':
                regularizer = activity_l1(l=regularizer_conf['value'])
            elif regularizer_conf['name'] == 'l2':
                regularizer = activity_l2(l=regularizer_conf['value'])
            elif regularizer_conf['name'] == 'l1l2':
                regularizer = activity_l1l2(l1=regularizer_conf['value'],
                                   l2=regularizer_conf['value'])
            model.add(Dense(dense_layers[i], activation=activation,
                             init=init, activity_regularizer=regularizer))
            model_structure += '-DN(' + str(dense_layers[i]) + ')'
            
            if dropout > 0:
                model.add(Dropout(dropout))
                model_structure += '-DR(' + str(dropout) + ')'
    last_regularizer = None
    if last_layer['regularization']['name'] == 'l1':
        last_regularizer = l1(l=last_layer['regularization']['value'])
    elif last_layer['regularization']['name'] == 'l2':
        last_regularizer = l2(l=last_layer['regularization']['value'])
    elif last_layer['regularization']['name'] == 'l1l2':
        last_regularizer = l1l2(l1=last_layer['regularization']['value'],
                           l2=last_layer['regularization']['value'])
    model.add(Dense(1, activation=last_layer['activation'],
                     W_regularizer=last_regularizer))
    model_structure += ('-LAST(' + last_layer['activation'] + ', '
                        + last_layer['regularization']['name'] + '('
                        + str(last_layer['regularization']['value'])
                              + '))|INIT:' + init +
                        '|REGULARIZATION:'
                        + str(regularizer_conf['name'])
                        + '(' + str(regularizer_conf['value']) + ')')
    
    compile_model(model, objective, optimization)
    # if optimization['name'] == 'adam':
    #     opt = Adam(lr=optimization['lr'], beta_1=0.9, beta_2=0.999,
    #                 epsilon=1e-08, decay=0.0)
    # else:
    #     opt = None
    # if objective == 'hinge':
    #     objective_fun = hinge_loss
    # else:
    #     objective_fun = objective
    # compile_model(model, loss=objective_fun,
    #               optimizer=opt, metrics=[
    #                                       # acc,
    #                                       # sensitivity,
    #                                       # specificity,
    #                                       partial_metrics
    #     ])
    model_structure += ('|OPTIMIZATION:' + optimization['name'] + '(lr='
                        + str(optimization['lr']) + ')'
                        + '|OBJECTIVE:' + objective['name'] + '('
                        + str(objective['penalty']) + ')'
                        + '|ACTIVATION:' + activation)
    
    return [model_structure, model]


# EOF

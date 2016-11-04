"""CNN for walking detection"""

# Authors: Julia Camps <julia.camps.sereix@est.fib.upc.edu>
# Created on: 06/10/2016 17:22

import numpy as np
import random as rd
import time

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.callbacks import Callback
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import MaxPooling1D

from FOG.preprocessing_tools import generate_arrays_from_file
from FOG.io_functions import get_patient_data_files
from FOG.io_functions import save_model
from FOG.io_functions import load_model
from FOG.preprocessing_tools import split_data
from FOG.models import build_model


_SEQ_CHANNEL = 1
_SEQ_FEATURE = 9
_N_CLASS = 2
_N_EPOCH = 50
_N_FOLD = 1
_T_WINDOW = 1
# _WINDOW_OVERLAP = 0.5
_SEQ_FREQ = 40
_DETECTION_PROBLEM = 'walk'
_PREPROCESS_FINISHED = True
_PRECALCULATE = False
_LOAD_MODEL = False
_TRAIN_MODEL = True
_TEST_MODEL = False
_BATCH_SIZE = 64
if _DETECTION_PROBLEM == 'fog':
    _N_TRAIN_SAMPLE = 338944
    _N_VAL_SAMPLE = 3136
    _N_TEST_SAMPLE = 5888
elif _DETECTION_PROBLEM == 'walk':
    _N_TRAIN_SAMPLE = 338944
    _N_VAL_SAMPLE = 3136
    _N_TEST_SAMPLE = 5888

_EARLY_STOPPING_TH = 0.01
_MAX_Q_SIZE = 5


class PrintBatch(Callback):
    """"""
    def on_batch_end(self, epoch, logs={}):
        """"""
        print(logs)


# def build_model():
#     """Build the model"""
#     model_ = Sequential()
#     window_size = _SEQ_FREQ * _T_WINDOW
#     model_.add(Convolution2D(64, 9, 9, border_mode='same',
#                              input_shape=(window_size,
#                                           _SEQ_FEATURE, _SEQ_CHANNEL),
#                              activation='relu'))
#     # model_.add(MaxPooling2D(pool_size=(2, 1)))
#     model_.add(Dropout(0.5))
#
#     model_.add(Convolution2D(64, 5, 1, activation='relu'))
#     # model_.add(MaxPooling2D(pool_size=(2, 1)))
#     model_.add(Dropout(0.5))
#
#     model_.add(Convolution2D(64, 3, 1, activation='relu'))
#     # model_.add(MaxPooling2D(pool_size=(2, 1)))
#     model_.add(Dropout(0.5))
#
#     # model_.add(Convolution2D(64, 3, 1, activation='relu'))
#     # # model_.add(MaxPooling2D(pool_size=(2, 1)))
#     # model_.add(Dropout(0.5))
#
#     model_.add(Flatten())
#     model_.add(Dense(128, activation='relu'))
#     model_.add(Dropout(0.5))
#
#     # model_.add(Dense(128, activation='relu'))
#     # model_.add(Dropout(0.5))
#
#     model_.add(Dense(1, activation='sigmoid'))
#
#     return model_


def load_trained_model(name='model'):
    """Load previously trained model

    Parameters
    ----------
    name : str, optional, default: 'model'

    Return
    ------
    model : keras.models.Sequential
    """
    model_ = build_model()
    return load_model(model_, name)


def train_model(model_, patient_list, type_name=_DETECTION_PROBLEM,
                cross_val=False, n_fold=_N_FOLD, n_epoch=_N_EPOCH,
                val_frac=0.1, stopping_th=_EARLY_STOPPING_TH):
    """Train model on the selected patients
    
    Parameters
    ----------
    model_ : keras.Sequential()
    patient_list : str array-like
        Names of the patients data to be used for training.
    cross_val : bool, optional, default: True
        Indicate if cross-validation strategy is to be used.
    n_fold : int, optional, default: _N_FOLD
        Number of folds, and of times to train the model over all
        the training data set.
    n_epoch : int, optional, default: _N_EPOCH
        Number of epochs to train the model.
    val_frac : float, optional, default: 0.1
        Proportion of data for validation. Only used if cross_val
        is False.
    
    Return
    ------
    model_ : keras.Sequential()
        Model trained with the specified data and configuration.
    result : dict
        Contains the resulting performance obtained during the
        training process.
        
    """
    
    if cross_val:
        [trained_model, result] = cross_validate(model_, n_fold)
    else:
        [train_patient, validation_patient] = split_data(
            patient_list, test=val_frac, random_=True,
            validation=True)
        train_file = [file for patient in train_patient for file in
                      get_patient_data_files(patient,
                                             type_name=type_name)]
        validation_file = [file for patient in validation_patient for
                           file in get_patient_data_files(
                               patient, type_name=type_name)]

        [trained_model, result] = single_train(
            model_, train_file, validation_file, n_epoch=n_epoch,
            stopping_th=stopping_th)
    
    return [trained_model, result]


def cross_validate(model_, n_fold=_N_FOLD, n_epoch=_N_EPOCH):
    """Perform cross-validation training strategy
    
    Parameters
    ----------
    model_ : keras.Sequential
    n_fold : int, optional, default: _N_FOLD
        Number of folds, and of times to train the model over all
        the training data set.
    n_epoch : int, optional, default: _N_EPOCH
        Number of epochs to train the model.
        
    Return
    ------
    model_ : keras.Sequential
        Model trained with the specified data and configuration.
    result : dict
        Contains the resulting performance obtained during the
        training process.
        
    """

    fold = gen_k_folds(train_patient)
    best_acc = 0
    best_model = None
    best_result = None
    result_cum = 0
    result = {}
    for i in range(n_fold):
        train_fold = (fold[:]).remove(i)
        test_fold = fold[i]
        [trained_model, result] = single_train(model_, train_fold,
                                               test_fold)
        if result['acc'] > best_acc:
            best_acc = result['acc']
            best_result = result
            best_model = trained_model
            
        result_cum += result['acc'] / n_fold
    result['val_acc'] = result_cum
    return [best_model, result]


def aux_generator():
    """"""
    while 1:
        yield (np.zeros((50,100,9,1)), np.zeros((50,1)))


def single_train(model_, train_file, validation_file,
                 n_epoch=_N_EPOCH, time_window=_T_WINDOW,
                 # window_overlaping=_WINDOW_OVERLAP,
                 data_freq=_SEQ_FREQ, stopping_th=_EARLY_STOPPING_TH):
    """Train the model
    
    
    """

    window_size = int(time_window * data_freq)
    # window_spacing = int(round(window_size * (1 - window_overlaping)))
    
    train_generator = generate_arrays_from_file(
        model_, train_file, window_size, #window_spacing,
        batch_size=_BATCH_SIZE, augment_count=[3, 7],
        augement_data_type='all')

    validation_generator = generate_arrays_from_file(
                model_, validation_file, window_size, #window_spacing,
                batch_size=_BATCH_SIZE)
    prev_acc = 0
    result_ = None
    acc = 0
    time_count = time.clock()
    epochs = 0
    status = 'OK'

    print('Entra')
    for i in range(n_epoch):
        result_ = [0, 0]
        try:
            model_.fit_generator(train_generator,
                                 samples_per_epoch=_N_TRAIN_SAMPLE,
                                 nb_epoch=1, verbose=1, callbacks=[],
                                 validation_data=None,
                                 nb_val_samples=None,
                                 class_weight=None,
                                 max_q_size=_MAX_Q_SIZE, nb_worker=1,
                                 pickle_safe=False)
        except Exception as e:
            status = 'ERROR:' + str(repr(e))
            break
        else:
            try:
                result_ = model_.evaluate_generator(
                    validation_generator, val_samples=_N_VAL_SAMPLE)
            except Exception as e:
                status = 'ERROR:' + str(repr(e))
                break
            else:
                epochs += 1
                print(result_)
                acc = result_[1]
                if (acc - prev_acc) < stopping_th or (1 - acc) < stopping_th:
                    print('Training Finished due to EARLY STOPPING')
                    break
                prev_acc = acc

    print('Train finished')
    print('Validation acc: ' + str(acc))
    return [model_, [result_, epochs, (time.clock() - time_count),
                     status]]


def test_model(model_, test_patient, time_window=_T_WINDOW,
                 # window_overlaping=_WINDOW_OVERLAP,
                 data_freq=_SEQ_FREQ, type_name=_DETECTION_PROBLEM):
    """Test the model


    """

    test_file = [file for patient in test_patient for file in
                  get_patient_data_files(patient,
                                         type_name=type_name)]
    
    window_size = int(time_window * data_freq)
    window_spacing = int(round(window_size * (1 - window_overlaping)))
    
    test_generator = generate_arrays_from_file(model_, test_file,
                                                window_size,
                                                window_spacing,
                                                batch_size=_BATCH_SIZE,
                                                augment_count=0)

    result_ = model.evaluate_generator(test_generator,
                                           val_samples=_N_VAL_SAMPLE)
    return result_


if __name__ == '__main__':
    
    from FOG.preprocessing_tools import full_preprocessing
    from FOG.preprocessing_tools import generate_dataset
    from FOG.preprocessing_tools import gen_k_folds
    
    # Get data
    [test_patient, train_patient] = generate_dataset()
    # Pre-calculate
    if _PRECALCULATE and not _PREPROCESS_FINISHED:
        full_preprocessing(train_patient,
                           type_name=_DETECTION_PROBLEM)

    # Initialize
    # seed = 170
    # seed = np.random.seed(seed)
    
    # Build model
    if _LOAD_MODEL:
        model = load_trained_model('model_fog')
    else:
        for j in range(10):
            window_size = _SEQ_FREQ * _T_WINDOW
            model = build_model(window_size)
            n_conv = [3, 3, 3, 4, 4, 4, 5, 5, 5]
            n_dense = [2, 2, 2, 2, 3, 3, 3, 3, 3]
            k_shapes = [[32, 15, 9], [32, 7, 1], [64, 5, 1], [64, 3, 1],
                        [128, 3, 1]]
            dense_shape = [128, 128, 256]
            pooling = [False, True, False, True, False, True, True,
                       True, True]
            dropout = [0.25, 0.5, 0.25, 0.5, 0.25, 0.5, 0.25, 0.5,
                       0.5]
            opt_name = ['adadelta', 'adam', 'rmsprop', 'adadelta', 'adam',
                        'rmsprop', 'adadelta', 'adam', 'rmsprop']
            for i in range(3, len(n_conv)):
                [conf_name, model] = build_model(
                    window_size, n_feature=_SEQ_FEATURE,
                    n_chan=_SEQ_CHANNEL, n_conv=n_conv[i],
                    n_dense=n_dense[i], k_shapes=k_shapes,
                    dense_shape=dense_shape,
                    opt_name=opt_name[i], pooling=pooling[i],
                    dropout=dropout[i])
                [model, result] = train_model(
                    model, train_patient,
                    type_name=_DETECTION_PROBLEM,
                    stopping_th=_EARLY_STOPPING_TH)
                with open("Output.txt", "a") as text_file:
                    print("\nConfiguration:\n" + conf_name +
                          '=>Epochs: ' +
                          str(result[1]) + ';Time: ' + str(result[2])
                          + ';Final Status: ' + result[3] + ';Acc: '
                          + str(result[0][1]),
                          file=text_file)
    
                save_model(model, 'model_' + str(i+j))

        
        
    #     # Train
    #     if _TRAIN_MODEL:
    #         train_it = 0
    #         [model, result] = train_model(model, train_patient,
    #                                       type_name=_DETECTION_PROBLEM,
    #                                       stopping_th=_EARLY_STOPPING_TH)
    #         # Save model
    #         # save_model(model, 'model_' + _DETECTION_PROBLEM + train_it)
    #         print('VALIDATION ACC: ' + str(result[1]))
    #
    # if _TEST_MODEL:
    #     result = test_model(model, test_patient,
    #                         time_window=_T_WINDOW,
    #                window_overlaping=_WINDOW_OVERLAP,
    #                data_freq=_SEQ_FREQ, type_name=_DETECTION_PROBLEM)
    #     print('TEST ACC: ' + str(result[1]))
    print('END')

# EOF

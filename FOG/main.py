"""CNN for walking detection"""

# Authors: Julia Camps <julia.camps.sereix@est.fib.upc.edu>
# Created on: 06/10/2016 17:22

import numpy as np
import random as rd
import time

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
_SEQ_FREQ = 40
_DETECTION_PROBLEM = 'walk'
_PREPROCESS_FINISHED = True
_PRECALCULATE = False
_LOAD_MODEL = False
_TRAIN_MODEL = True
_TEST_MODEL = False
_BATCH_SIZE = 64
_N_TRAIN_SAMPLE = 537600
_N_VAL_SAMPLE = 10368
_N_TEST_SAMPLE = 19392
_PERCENTAGE_OF_POSITIVE_TRAIN = 0.581491815476
_PERCENTAGE_OF_POSITIVE_VAL = 0.550636574074
_PERCENTAGE_OF_POSITIVE_TEST = 0.581941006601
_EARLY_STOPPING_TH = 0.01
_MAX_Q_SIZE = 5
_REPRODUCIBILITY = True
_SEED = 277
_N_SHIFT = 2
_N_ROTATE = 4


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
                n_epoch=_N_EPOCH, val_frac=0.1,
                stopping_th=_EARLY_STOPPING_TH):
    """Train model on the selected patients
    
    Parameters
    ----------
    model_ : keras.Sequential()
    patient_list : str array-like
        Names of the patients data to be used for training.
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


def single_train(model_, train_file, validation_file,
                 n_epoch=_N_EPOCH, time_window=_T_WINDOW,
                 data_freq=_SEQ_FREQ, stopping_th=_EARLY_STOPPING_TH):
    """Train the model
    
    
    """

    window_size = int(time_window * data_freq)
    
    train_generator = generate_arrays_from_file(
        model_, train_file, window_size, batch_size=_BATCH_SIZE,
        augment_count=[_N_SHIFT, _N_ROTATE], augement_data_type='all')

    validation_generator = generate_arrays_from_file(
                model_, validation_file, window_size,
                batch_size=_BATCH_SIZE)
    prev_acc = 0
    result_ = None
    acc = 0
    time_count = time.clock()
    epochs = 0
    status = 'OK'
    
    print('TRAINING MODEL: ' + model.conf_name)

    for i in range(n_epoch):
        print('\n==================\nSTARTING EPOCH: ' + str(i)
              + '\n========================')
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
        print('==================\nENDED EPOCH: ' + str(i)
              + '\n========================')
    print('TRAINING FINISHED WITH VALIDATION ACC: ' + str(acc))
    return [model_, [result_, epochs, (time.clock() - time_count),
                     status]]


def test_model(model_, test_patient, time_window=_T_WINDOW,
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
    
    # Get data
    [test_patient, train_patient] = generate_dataset()
    # Pre-calculate
    if _PRECALCULATE and not _PREPROCESS_FINISHED:
        full_preprocessing(train_patient,
                           type_name=_DETECTION_PROBLEM)

    # Initialize
    if _REPRODUCIBILITY:
        np.random.seed(_SEED)
        rd.seed(_SEED)
    
    # Build model
    if _LOAD_MODEL:
        model = load_trained_model('model_fog')
    else:
        initializations = ['lecun_uniform', 'glorot_normal',
                           'glorot_uniform', 'he_normal',
                           'he_uniform']
        n_conv = [2, 2, 3, 3, 3, 4, 4, 4, 5]
        n_dense = [1, 2, 2, 2, 2, 3, 3, 3, 3]
        k_shapes = [[32, 7], [32, 5], [64, 3], [64, 3],
                    [128, 3]]
        dense_shape = [128, 128, 256]
        pooling = [False, True, False, True, False, True, True,
                   True, True]
        dropout = [0.25, 0.5, 0.25, 0.5, 0.25, 0.5, 0.25, 0.5,
                   0.5]
        opt_name = ['adadelta', 'adam', 'rmsprop', 'adadelta',
                    'adam', 'rmsprop', 'adadelta', 'adam',
                    'rmsprop']
        window_size = _SEQ_FREQ * _T_WINDOW
        for j in range(len(initializations)):
            model = build_model(window_size)
            init = initializations[j]
            for i in range(len(n_conv)):
                [conf_name, model] = build_model(
                    window_size, n_feature=_SEQ_FEATURE,
                    n_conv=n_conv[i], n_dense=n_dense[i],
                    k_shapes=k_shapes, dense_shape=dense_shape,
                    opt_name=opt_name[i], pooling=pooling[i],
                    dropout=dropout[i], init=init)
                model.conf_name = conf_name
                [model, result] = train_model(
                    model, train_patient,
                    type_name=_DETECTION_PROBLEM,
                    stopping_th=_EARLY_STOPPING_TH)
                with open("Output.txt", "a") as text_file:
                    print("\nConfiguration:\n" + conf_name +
                          '\nEpochs: ' +
                          str(result[1]) + '\nTime: ' + str(result[2])
                          + '\nFinal Status: ' + result[3]
                          + '\nValidation acc: ' + str(result[0][1])
                          + '\nPercentage of positive train samples: '
                          + str(_PERCENTAGE_OF_POSITIVE_TRAIN)
                          + '\nPercentage of positive val samples: '
                          + str(_PERCENTAGE_OF_POSITIVE_VAL)
                          + '\nNumber of training samples: '
                          + str(_N_TRAIN_SAMPLE)
                          + '\n Number of model parameters: ' +
                          str(model.count_params()),
                          file=text_file)
    
                save_model(model, 'model_' + str(i + (j*len(n_conv))))

    print('END')

# EOF

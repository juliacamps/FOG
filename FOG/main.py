"""CNN for walking detection"""

# Authors: Julia Camps <julia.camps.sereix@est.fib.upc.edu>
# Created on: 06/10/2016 17:22

import numpy as np
import random as rd
import time

from keras.callbacks import EarlyStopping

from FOG.preprocessing_tools import generate_arrays_from_file
from FOG.preprocessing_tools import confusion_matrix
from FOG.io_functions import get_patient_data_files
from FOG.io_functions import save_model
from FOG.io_functions import load_model
from FOG.models import build_model


_SEQ_CHANNEL = 1
_SEQ_FEATURE = 9
_N_CLASS = 2
_N_EPOCH = 30
_N_FOLD = 1
_SEQ_FREQ = 100
_DETECTION_PROBLEM = 'fog'
_PREPROCESS_FINISHED = True
_PRECALCULATE = False
_LOAD_MODEL = False
_TRAIN_MODEL = True
_TEST_MODEL = False
_PREPROCESSING_CONF = {
                       # 'Conf_small_raw': {'Conf': {'Time': 1,
                       #                             'Batch': 128,
                       #                             'Filter': 0.0},
                       #                    'Result': {
                       #                       'n_train': 549248,
                       #                       'n_val': 10624,
                       #                       'n_test': 19712,
                       #                       'percent_train': 0.59,
                       #                       'percent_val': 0.56,
                       #                       'percent_test': 0.59}
                       #                    },
                       # 'Conf_small_low': {'Conf': {'Time': 1,
                       #                             'Batch': 128,
                       #                             'Filter': 0.1},
                       #                    'Result': {
                       #                        'n_train': 547584,
                       #                        'n_val': 10624,
                       #                        'n_test': 19584,
                       #                        'percent_train': 0.59,
                       #                        'percent_val': 0.56,
                       #                        'percent_test': 0.58}
                       #                    },
                       # 'Conf_small_med': {'Conf': {'Time': 1,
                       #                             'Batch': 128,
                       #                             'Filter': 0.2},
                       #                    'Result': {
                       #                        'n_train': 545664,
                       #                        'n_val': 10624,
                       #                        'n_test': 19584,
                       #                        'percent_train': 0.59,
                       #                        'percent_val': 0.56,
                       #                        'percent_test': 0.58}
                       #                    },
                       # 'Conf_small_high': {'Conf': {'Time': 1,
                       #                              'Batch': 128,
                       #                              'Filter': 0.3},
                       #                     'Result': {
                       #                        'n_train': 543488,
                       #                        'n_val': 10496,
                       #                        'n_test': 19456,
                       #                        'percent_train': 0.59,
                       #                        'percent_val': 0.56,
                       #                        'percent_test': 0.58}
                       #                     },
                       # 'Conf_med_raw': {'Conf': {'Time': 2,
                       #                           'Batch': 64,
                       #                           'Filter': 0.0},
                       #                  'Result': {
                       #                        'n_train': 274560,
                       #                        'n_val': 5312,
                       #                        'n_test': 9856,
                       #                        'percent_train': 0.61,
                       #                        'percent_val': 0.58,
                       #                        'percent_test': 0.60}
                       #                  },
                       # 'Conf_med_low': {'Conf': {'Time': 2,
                       #                           'Batch': 64,
                       #                           'Filter': 0.1},
                       #                  'Result': {
                       #                        'n_train': 272896,
                       #                        'n_val': 5312,
                       #                        'n_test': 9792,
                       #                        'percent_train': 0.60,
                       #                        'percent_val': 0.58,
                       #                        'percent_test': 0.60}
                       #                  },
                       # 'Conf_med_med': {'Conf': {'Time': 2,
                       #                           'Batch': 64,
                       #                           'Filter': 0.2},
                       #                  'Result': {
                       #                        'n_train': 270976,
                       #                        'n_val': 5248,
                       #                        'n_test': 9728,
                       #                        'percent_train': 0.60,
                       #                        'percent_val': 0.58,
                       #                        'percent_test': 0.60}
                       #                  },
                       # 'Conf_med_high': {'Conf': {'Time': 2,
                       #                            'Batch': 64,
                       #                            'Filter': 0.3},
                       #                   'Result': {
                       #                        'n_train': 268864,
                       #                        'n_val': 5184,
                       #                        'n_test': 9664,
                       #                        'percent_train': 0.60,
                       #                        'percent_val': 0.57,
                       #                        'percent_test': 0.60}
                       #                   },
                       # 'Conf_high_raw': {'Conf': {'Time': 3,
                       #                            'Batch': 32,
                       #                            'Filter': 0.0},
                       #                   'Result': {
                       #                        'n_train': 182912,
                       #                        'n_val': 3552,
                       #                        'n_test': 6560,
                       #                        'percent_train': 0.62,
                       #                        'percent_val': 0.60,
                       #                        'percent_test': 0.62}
                       #                   },
                       # 'Conf_high_low': {'Conf': {'Time': 3,
                       #                            'Batch': 32,
                       #                            'Filter': 0.1},
                       #                   'Result': {
                       #                        'n_train': 181408,
                       #                        'n_val': 3520,
                       #                        'n_test': 6496,
                       #                        'percent_train': 0.62,
                       #                        'percent_val': 0.59,
                       #                        'percent_test': 0.62}
                       #                   },
                       # 'Conf_high_med': {'Conf': {'Time': 3,
                       #                            'Batch': 32,
                       #                            'Filter': 0.2},
                       #                   'Result': {
                       #                        'n_train': 179872,
                       #                        'n_val': 3488,
                       #                        'n_test': 6432,
                       #                        'percent_train': 0.62,
                       #                        'percent_val': 0.59,
                       #                        'percent_test': 0.61}
                       #                   },
                       # 'Conf_high_high': {'Conf': {'Time': 3,
                       #                             'Batch': 32,
                       #                             'Filter': 0.3},
                       #                    'Result': {
                       #                        'n_train': 177568,
                       #                        'n_val': 3456,
                       #                        'n_test': 6368,
                       #                        'percent_train': 0.61,
                       #                        'percent_val': 0.59,
                       #                        'percent_test': 0.61}
                       #                    },
                       # 'Conf_2_4': {'Conf': {'Time': 2,
                       #                       'Batch': 64,
                       #                       'Filter': 0.4
                       #                       },
                       #              'Result': {'n_train': 266816,
                       #                         'n_val': 9536,
                       #                         'n_test': 5120,
                       #                         'percent_train': 0.60,
                       #                         'percent_val': 0.59,
                       #                         'percent_test': 0.57
                       #                         }
                       #              },
                       # 'Conf_2_5': {'Conf': {'Time': 2,
                       #                       'Batch': 64,
                       #                       'Filter': 0.5
                       #                       },
                       #              'Result': {'n_train': 185984,
                       #                         'n_val': 6400,
                       #                         'n_test': 3968,
                       #                         'percent_train': 0.14,
                       #                         'percent_val': 0.13,
                       #                         'percent_test': 0.24
                       #                         }
                       #              },
                       'Conf_3_5': {'Conf': {'Time': 3,
                                             'Batch': 32,
                                             'Filter': 0.5
                                             },
                                    'Result': {'n_train': 122304,
                                               'n_val': 4160,
                                               'n_test': 2560,
                                               'percent_train': 0.15,
                                               'percent_val': 0.13,
                                               'percent_test': 0.24
                                               }
                                    }#,
                       # 'Conf_2_5': {'Conf': {'Time': 2,
                       #                       'Batch': 64,
                       #                       'Filter': 0.5
                       #                       },
                       #              'Result': {'n_train': 184960,
                       #                         'n_val': 6336,
                       #                         'n_test': 3968,
                       #                         'percent_train': 0.41,
                       #                         'percent_val': 0.39,
                       #                         'percent_test': 0.44
                       #                         }
                       #              }#,
    
                       # 'Conf_2_6': {'Conf': {'Time': 2,
                       #                       'Batch': 64,
                       #                       'Filter': 0.6
                       #                       },
                       #              'Result': {'n_train': 262848,
                       #                         'n_val': 9408,
                       #                         'n_test': 5056,
                       #                         'percent_train': 0.59,
                       #                         'percent_val': 0.59,
                       #                         'percent_test': 0.56
                       #                         }
                       #              }#,
                       # 'Conf_2_7': {'Conf': {'Time': 2,
                       #                       'Batch': 64,
                       #                       'Filter': 0.7
                       #                       },
                       #              'Result': {'n_train': 260800,
                       #                         'n_val': 9344,
                       #                         'n_test': 4992,
                       #                         'percent_train': 0.59,
                       #                         'percent_val': 0.59,
                       #                         'percent_test': 0.56
                       #                         }
                       #              }
                       }

_EARLY_STOPPING_TH = 0.01
_MAX_Q_SIZE = 5
_REPRODUCIBILITY = True
_SEED = 77
_N_SHIFT = 2
_N_ROTATE = 4
_MIN_EPOCH = 10


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


def train_model(model_, train_patient, validation_patient=None,
                type_name=_DETECTION_PROBLEM, n_epoch=_N_EPOCH,
                stopping_th=_EARLY_STOPPING_TH, n_train_sample=0,
                n_val_sample=0, filter_threshold=0.2,
                batch_size=32, window_size=_SEQ_FREQ):
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
    
    train_file = [file for patient in train_patient for file in
                  get_patient_data_files(patient,
                                         type_name=type_name)]
    validation_file = None
    if validation_patient is not None:
        validation_file = [file for patient in validation_patient for
                           file in get_patient_data_files(
                               patient, type_name=type_name)]

    [trained_model, result] = single_train(
        model_, train_file, validation_file, n_epoch=n_epoch,
        stopping_th=stopping_th, n_train_sample=n_train_sample,
        n_val_sample=n_val_sample, batch_size=batch_size,
        window_size=window_size, filter_threshold=filter_threshold)
    
    return [trained_model, result]


def single_train(model_, train_file, validation_file=None,
                 n_epoch=_N_EPOCH, stopping_th=_EARLY_STOPPING_TH,
                 n_train_sample=0, n_val_sample=0, batch_size=32,
                 window_size=_SEQ_FREQ, filter_threshold=0.2):
    """Train the model
    
    
    """
    
    train_generator = generate_arrays_from_file(
        train_file, window_size, batch_size=batch_size,
        augment_count=[_N_SHIFT, _N_ROTATE],
        augement_data_type='all', filter_threshold=filter_threshold)

    validate = False
    validation_generator = None
    if validation_file is not None:
        validation_generator = generate_arrays_from_file(
            validation_file, window_size,
            batch_size=batch_size, filter_threshold=filter_threshold)
        validate = True

    acc = 0
    time_count = time.clock()
    epochs = 0
    status = 'OK'
    result_ = np.zeros(6)

    abort_train = False
    for epoch_it in range(n_epoch):
        print('\n==================\nSTARTING EPOCH: ' + str(epoch_it)
              + '\n========================')
        end_epoch = False
        samples_epoch_count = 0
        # Notice that the following would be an endless loop
        # without the counter exit condition
        for X, y in train_generator:
            try:
                model_.train_on_batch(X, y, class_weight={0: 1,
                                                          1: 1000})
            except Exception as e:
                status = ('ERROR: While training the model: '
                          + str(repr(e)))
                end_epoch = True
                abort_train = True
            else:
                samples_epoch_count += batch_size
                
            if (samples_epoch_count + 1) >= n_train_sample:
                end_epoch = True
            
            if end_epoch:
                # End loop condition (only condition)
                break
        if status == 'OK':
            [status, train_conf_mat] = get_conf_mat(
                model_, train_generator, n_train_sample, batch_size,
                msg='train error prediction')
            if status != 'OK':
                abort_train = True
                
            elif validate:
                [status, val_conf_mat] = get_conf_mat(
                    model_, validation_generator, n_val_sample,
                    batch_size, msg='validation error prediction')
                
                if status != 'OK':
                    abort_train = True
                    
            else:
                print('Validation is not configured')

        if abort_train:
            # An error occurred while training -> Abort and start next
            break
        else:
            epochs += 1
            
            print('Train conf-mat:')
            print(train_conf_mat)
            [ACC, PPV, TPR] = acc_precision_recall(train_conf_mat)
            print('ACC: ' + str(ACC))
            print('Train precision: ' + str(PPV))
            print('Train recall: ' + str(TPR))
            result_[0] = ACC
            result_[1] = PPV
            result_[2] = TPR
            
            print('Validation conf-mat:')
            print(val_conf_mat)
            [ACC, PPV, TPR] = acc_precision_recall(val_conf_mat)
            print('ACC: ' + str(ACC))
            print('Validation precision: ' + str(PPV))
            print('Validation recall: ' + str(TPR))
            result_[3] = ACC
            result_[4] = PPV
            result_[5] = TPR
            
        print('==================\nENDED EPOCH: ' + str(epoch_it)
              + '\n========================')
            
    if not abort_train:
        print('TRAINING FINISHED WITH VALIDATION ACC: ' + str(acc))
    else:
        print('TRAINING WAS ABORTED')
    return [model_, [result_, epochs, (time.clock() - time_count),
                     status]]


def acc_precision_recall(conf_mat):
    """"""
    TP = conf_mat[0, 0]
    FN = conf_mat[0, 1]
    FP = conf_mat[1, 0]
    TN = conf_mat[1, 1]
    ACC = (TP + TN) / (TP + FN + FP + TN)
    PPV = TP / (TP + FP)
    TPR = TP / (TP + FN)
    return [ACC, PPV, TPR]


def get_conf_mat(model_, generator, samples_count, batch_size, msg):
    """"""
    
    conf_mat = np.zeros((2, 2))
    status = 'OK'
    end_epoch = False
    samples_it = 0
    for X, y_true in generator:
        try:
            y_pred = model_.predict_on_batch(X)
        except Exception as e:
            status = ('ERROR: While ' + msg + ': ' + str(repr(e)))
            end_epoch = True
        else:
            conf_mat += confusion_matrix(y_true, y_pred)
            samples_it += batch_size
        if samples_it >= samples_count:
            end_epoch = True
    
        if end_epoch:
            # End loop condition (only condition)
            break
    return [status, conf_mat]


if __name__ == '__main__':
    
    import datetime
    
    from FOG.preprocessing_tools import full_preprocessing
    from FOG.preprocessing_tools import generate_dataset
    
    # Get data
    [train_patient, val_patient, test_patient] = generate_dataset()
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
        model = load_trained_model('model_0')
    else:
        initializations = ['glorot_normal',
                           'glorot_uniform', 'lecun_uniform',
                           'he_normal', 'he_uniform']
        n_convs = [3]
        n_denses = [1]
        k_shape = [[32, 11], [32, 5], [64, 3]]
        dense_shape = [128]
        dropouts = [0.0, 0.5]
        opt_names = ['adadelta']
        poolings = [False, True]
        atrous = [False, True]
        mod_id = 0
        for init in initializations:
            for n_conv in n_convs:
                for n_dense in n_denses:
                    for pooling in poolings:
                        for dropout in dropouts:
                            for atrou in atrous:
                                for opt_name in opt_names:
                                    for key, conf in \
                                            _PREPROCESSING_CONF.items():
                                        window_time = conf['Conf']['Time']
                                        window_size = (_SEQ_FREQ
                                                       * window_time)
                                        batch_size = conf['Conf']['Batch']
                                        filter_th = conf['Conf']['Filter']
                                        n_train = conf['Result']['n_train']
                                        n_val = conf['Result']['n_val']
                                        percent_pos_train = conf['Result'][
                                            'percent_train']
                                        percent_pos_val = conf['Result'][
                                            'percent_val']
                                        [conf_name, model] = build_model(
                                            window_size,
                                            n_feature=_SEQ_FEATURE,
                                            n_conv=n_conv, n_dense=n_dense,
                                            k_shapes=k_shape,
                                            dense_shape=dense_shape,
                                            opt_name=opt_name,
                                            pooling=pooling,
                                            dropout=dropout,
                                            init=init,
                                            atrous=atrou)
                                        print('STARTING TRAINING OF '
                                              'MODEL:\n' + conf_name +
                                              '\nN-Parameters='
                                              + str(model.count_params())
                                              + '\nConf: ' + key)
                                        [model, result] = train_model(
                                            model, train_patient,
                                            validation_patient=val_patient,
                                            type_name=_DETECTION_PROBLEM,
                                            stopping_th=_EARLY_STOPPING_TH,
                                            n_train_sample=n_train,
                                            n_val_sample=n_val,
                                            filter_threshold=filter_th,
                                            batch_size=batch_size,
                                            window_size=window_size)
                                        with open('Output.log', 'a') as \
                                                out_f:
                                            print('\nDate:'
                                                  + str(datetime.date.today())
                                                  + '\nConfiguration: '
                                                  + key + '\n'
                                                  + conf_name + '\nEpochs: '
                                                  + str(result[1])
                                                  + '\nTime: '
                                                  + str(result[2])
                                                  + '\nFinal Status: '
                                                  + result[3]
                                                  + '\nWidow time in sec: '
                                                  + str(window_time)
                                                  + '\nNumber of training '
                                                  + 'samples: '
                                                  + str(n_train)
                                                  + '\nNumber of model '
                                                  + 'parameters: '
                                                  + str(model.count_params())
                                                  + '\nModel-ID: '
                                                  + str(mod_id)
                                                  + '\nTRAIN RESULTS'
                                                  + '\n  Percentage of '
                                                  + _DETECTION_PROBLEM
                                                  + ' samples: '
                                                  + str(percent_pos_train)
                                                  + '\n  Train acc: '
                                                  + str(result[0][0])
                                                  + '\n  Train precision: '
                                                  + str(result[0][1])
                                                  + '\n  Train recall: '
                                                  + str(result[0][2])
                                                  + '\nVALIDATION RESULTS'
                                                  + '\n  Percentage of '
                                                  + _DETECTION_PROBLEM
                                                  + ' samples: '
                                                  + str(percent_pos_train)
                                                  + '\n  Validation acc: '
                                                  + str(result[0][3])
                                                  + '\n  Validation '
                                                  + 'precision: '
                                                  + str(result[0][4])
                                                  + '\n  Validation recall: '
                                                  + str(result[0][5]),
                                                  file=out_f)
                                            mod_id += 1
                                            save_model(model,
                                                       'model_' + str(mod_id))

    print('END')

# EOF

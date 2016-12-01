"""CNN for walking detection"""

# Authors: Julia Camps <julia.camps.sereix@est.fib.upc.edu>
# Created on: 06/10/2016 17:22

import numpy as np
import random as rd
import time

from FOG.preprocessing_tools import generate_arrays_from_file
from FOG.preprocessing_tools import get_generator
from FOG.utils import calc_batch_size
from FOG.io_functions import save_model
from FOG.io_functions import load_model
from FOG.io_functions import get_next_id
from FOG.io_functions import record_settings_results
from FOG.models import build_model
from FOG.metrics import get_statistics
from FOG.metrics import metrics_calc
from FOG.definitions import _get_patient


_DETECTION_PROBLEM = 'fog'
_GENERATE_SUMMARY = True
_TRAIN_MODEL = True
_TEST_MODEL = False
_REPRODUCIBILITY = True
_N_EPOCH = 200  # x2


def train_model(model_, train_patient, validation_patient=None,
                problem=_DETECTION_PROBLEM, n_epoch=_N_EPOCH,
                n_train_sample=0, n_val_sample=0,
                filter_threshold=0.2, batch_size=64,
                window_size=200):
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

    [train_file, validation_file] = get_train_validation_path(
        train_patient, validation_patient, problem=problem)

    train_generator = generate_arrays_from_file(
        train_file, window_size, batch_size=batch_size,
        augment_count=[_N_SHIFT, _N_ROTATE],
        filter_threshold=filter_threshold)

    if validation_file is not None:
        validation_generator = generate_arrays_from_file(
            validation_file, window_size, batch_size=batch_size,
            filter_threshold=filter_threshold)
    else:
        validation_generator = None

    [trained_model, result] = single_train(
        model_, train_generator,
        validation_generator=validation_generator, n_epoch=n_epoch,
        n_train_sample=n_train_sample, n_val_sample=n_val_sample,
        batch_size=batch_size)
    
    return [trained_model, result]


def single_train(model_, train_generator, validation_generator=None,
                 n_epoch=_N_EPOCH, n_train_sample=0,
                 n_val_sample=0, batch_size=64):
    """Train the model
    
    
    """

    acc = 0
    time_count = time.clock()
    epochs = 0
    status = 'OK'
    result_ = np.zeros(6)
    train_acc = 0
    train_sensitivity = 0
    train_specificity = 0
    validation_acc = 0
    validation_sensitivity = 0
    validation_specificity = 0
    A_E_statistics = []

    abort_train = False
    for epoch_it in range(n_epoch):
        print('\n==================\nSTARTING EPOCH: ' + str(epoch_it)
              + '\n========================')
        end_epoch = False
        samples_epoch_count = 0
        # Notice that the following would be an endless loop
        # without the counter exit condition
        for X, y, y_orig in train_generator:
            try:
                model_.train_on_batch(X, y)
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
            [status, train_conf_mat, A_E_statistics_train] = \
                get_statistics(model_, train_generator,
                               n_train_sample, batch_size,
                               msg='train error prediction')
            if status != 'OK':
                abort_train = True
            else:
                A_E_statistics.append({'train': {'partial_results':
                    A_E_statistics_train}})
                
                if validation_generator is not None:
                    [status, val_conf_mat, A_E_statistics_val] = \
                        get_statistics(model_, validation_generator,
                                       n_val_sample, batch_size,
                                       msg=
                                       'validation error prediction')
                    
                    if status != 'OK':
                        abort_train = True
                    else:
                        A_E_statistics[epoch_it]['validation'] = (
                            {'partial_results': A_E_statistics_val})
                    
                else:
                    print('Validation is not configured')

        if abort_train:
            # An error occurred while training -> Abort and start next
            break
        else:

            # print metrics
            print('Train conf-mat:')
            print(train_conf_mat)
            metics = metrics_calc(train_conf_mat,
                                  metrics=['accuracy', 'sensitivity',
                                           'specificity'])
            train_acc = metics['accuracy']
            train_sensitivity = metics['sensitivity']
            train_specificity = metics['specificity']
            print('ACC: ' + str(train_acc))
            print('Train sensitivity: ' + str(train_sensitivity))
            print('Train specificity: ' + str(train_specificity))

            print('Validation conf-mat:')
            print(val_conf_mat)
            metics = metrics_calc(val_conf_mat,
                                  metrics=['accuracy', 'sensitivity',
                                           'specificity'])
            validation_acc = metics['accuracy']
            validation_sensitivity = metics['sensitivity']
            validation_specificity = metics['specificity']
            print('ACC: ' + str(validation_acc))
            print('Validation sensitivity: '
                  + str(validation_sensitivity))
            print('Validation specificity: '
                  + str(validation_specificity))
            
            # Store epoch metrics
            A_E_statistics[epoch_it]['train']['global_results'] = {
                'conf_mat': train_conf_mat, 'accuracy': train_acc,
                'sensitivity': train_sensitivity,
                'specificity': train_specificity
            }
            A_E_statistics[epoch_it]['validation'][
                'global_results'] = {'conf_mat': val_conf_mat,
                                     'accuracy': validation_acc,
                                     'sensitivity': validation_sensitivity,
                                     'specificity': validation_specificity
                                     }
            # [status, prediction] = predict_single_result()
            epochs += 1
        print('==================\nENDED EPOCH: ' + str(epoch_it)
              + '\n========================')
            
    if not abort_train:
        print('TRAINING FINISHED WITH VALIDATION ACC: ' + str(acc))
    else:
        print('TRAINING WAS ABORTED')
    result_[0] = train_acc
    result_[1] = train_sensitivity
    result_[2] = train_specificity
    result_[3] = validation_acc
    result_[4] = validation_sensitivity
    result_[5] = validation_specificity
    
    return [model_, [result_, epochs, (time.clock() - time_count),
                     status, A_E_statistics]]


def predict_result(
        model, train_patient, validation_patient, problem, n_train,
        n_val, threshold, batch_size, augment_shift, window_size,
        augment_rotate, normalize):
    """"""
    [train_generator, validation_generator] = get_generator(
        train_patient, validation_patient=validation_patient,
        problem=problem, augment_shift=augment_shift,
        augment_rotate=augment_rotate, threshold=threshold,
        normalize=normalize, window_size=window_size,
        batch_size=batch_size)
    
    return predict_single_result(model, train_generator,
                                 validation_generator, n_train,
                                 n_val, batch_size)


def predict_single_result(model, train_generator,
                          validation_generator=None, n_train=0,
                          n_val=0, batch_size=64):
    """"""
    result = {}
    [status, train_conf_mat, train_statistic] = \
        get_statistics(model, train_generator, n_train, batch_size,
                       msg='train_error_reproduction')
    if status == 'OK':
        metics = metrics_calc(
            train_conf_mat, metrics=['accuracy', 'sensitivity',
                                     'specificity'])
        train_acc = metics['accuracy']
        train_sensitivity = metics['sensitivity']
        train_specificity = metics['specificity']
        
        result['train'] = {
            'partial_result': train_statistic,
            'global_result': {
                'conf_mat': train_conf_mat,
                'accuracy': train_acc,
                'sensitivity': train_sensitivity,
                'specificity': train_specificity
            }
        }
        if validation_generator is not None:
            [status, val_conf_mat, validation_statistic] = \
                get_statistics(model, validation_generator,
                               n_val, batch_size,
                               msg=
                               'validation_error_reproduction')
            if status == 'OK':
                metics = metrics_calc(
                    val_conf_mat, metrics=['accuracy',
                                           'sensitivity',
                                           'specificity'])
                
                validation_acc = metics['accuracy']
                validation_sensitivity = metics['sensitivity']
                validation_specificity = metics['specificity']
                
                result['validation'] = {
                    'partial_result': validation_statistic,
                    'global_result': {
                        'conf_mat': val_conf_mat,
                        'accuracy': validation_acc,
                        'sensitivity': validation_sensitivity,
                        'specificity': validation_specificity
                    }
                }
                          
    return [status, result]


if __name__ == '__main__':
    
    
    
    from FOG.preprocessing_tools import full_preprocessing
    from FOG.preprocessing_tools import generate_dataset
    from FOG.io_functions import get_models
    from FOG.io_functions import save_prediction_result
    
    # Get data
    [train_patient, val_patient, test_patient] = generate_dataset()

    # Initialize
    if _REPRODUCIBILITY:
        np.random.seed(_SEED)
        rd.seed(_SEED)

    # Build model
    if _TRAIN_MODEL:
        initializations = ['glorot_uniform', 'lecun_uniform',
                           'he_normal', 'he_uniform', 'glorot_normal']
        n_convs = [4]
        n_denses = [1]
        k_shape = [[32, 11], [32, 5], [64, 3], [64, 3]]
        dense_shape = [128]
        dropouts = [0.25, 0.5]
        opt_names = ['adadelta']
        poolings = [True, False]
        atrous = [True, False]
        regularizers = [None, 'l1', 'l2']
        mod_id = get_next_id()
        day_date = str(datetime.date.today())
        with open('Output.log', 'a') as out_f:
            print('\n\n============= NEW EXPERIMENT ============='
                  + '\nStarting date: ' + day_date
                  + '\nStarting model ID: ' + mod_id + '\n',
                  file=out_f)
        for init in initializations:
            for n_conv in n_convs:
                for n_dense in n_denses:
                    pooling = poolings[0]
                    # for pooling in poolings:
                    dropout = dropouts[0]
                    # for dropout in dropouts:
                    atrou = atrous[0]
                    # for atrou in atrous:
                    regularizer = regularizers[0]
                    # for regularizer in regularizers:
                    for opt_name in opt_names:
                        for conf_key, conf in \
                                _PREPROCESSING_CONF.items():
                            window_time = conf['Conf']['Time']
                            window_size = _SEQ_FREQ * window_time
                            batch_size = calc_batch_size(
                                _MAX_BATCH_SIZE, window_time)
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
                                atrous=atrou,
                                regularizer=regularizer)
                            model_name = 'model_' + str(mod_id)
                            day_date = str(datetime.date.today())
                            nb_parameters = model.count_params()
                            
                            settings = {'date': day_date,
                                        'augment_shift': _N_SHIFT,
                                        'augment_rotate': _N_ROTATE,
                                        'random_seed': _SEED,
                                        'train_patient':
                                            train_patient,
                                        'val_patient': val_patient,
                                        'normalize': _NORMALIZE,
                                        'problem': _DETECTION_PROBLEM,
                                        'model_name': model_name,
                                        'conf_name': conf_key,
                                        'model_conf': conf_name,
                                        'window_time': window_time,
                                        'batch_size': batch_size,
                                        'n_epoch': _N_EPOCH,
                                        'n_parameter': nb_parameters,
                                        'dropout': dropout,
                                        'regularization': regularizer,
                                        'atrous': atrou,
                                        'pooling': pooling,
                                        'initialization': init,
                                        'optimizer': opt_name,
                                        'n_train': n_train,
                                        'percent_pos_train':
                                            percent_pos_train,
                                        'n_validation': n_val,
                                        'percent_pos_val':
                                            percent_pos_val,
                                        'threshold': filter_th,
                                        'window_size': window_size
                                        }
                            print('STARTING TRAINING OF MODEL:\n'
                                  + conf_name + '\nN-Parameters='
                                  + str(model.count_params())
                                  + '\nConf: ' + conf_key)
                            [model, result] = train_model(
                                model, train_patient,
                                validation_patient=val_patient,
                                problem=_DETECTION_PROBLEM,
                                n_train_sample=n_train,
                                n_val_sample=n_val,
                                filter_threshold=filter_th,
                                batch_size=batch_size,
                                window_size=window_size)
                            train_acc = str(result[0][0])
                            train_sensitivity = str(result[0][1])
                            train_specificity = str(result[0][2])
                            validation_acc = str(result[0][3])
                            validation_sensitivity = str(result[0][4])
                            validation_specificity = str(result[0][5])
                            completed_epochs = result[1]
                            training_time = result[2]
                            final_status = result[3]
                            settings['train_acc'] = train_acc
                            settings['train_sensitivity'] = \
                                train_sensitivity
                            settings['train_specificity'] = \
                                train_specificity
                            settings['validation_acc'] = \
                                validation_acc
                            settings['validation_sensitivity'] = \
                                validation_sensitivity
                            settings['validation_specificity'] = \
                                validation_specificity
                            settings['n_completed_epoch'] = \
                                completed_epochs
                            settings['training_time'] = training_time
                            settings['final_status'] = final_status
                            settings['train_log'] = result[4]
                            
                            with open('Output.log', 'a') as out_f:
                                print('\nDate:'
                                      + day_date
                                      + '\nConfiguration: '
                                      + conf_key + '\n'
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
                                      + str(nb_parameters)
                                      + '\nModel-Name-ID: '
                                      + model_name
                                      + '\nTRAIN RESULTS'
                                      + '\n  Percentage of '
                                      + _DETECTION_PROBLEM
                                      + ' samples: '
                                      + str(percent_pos_train)
                                      + '\n  Train acc: '
                                      + str(result[0][0])
                                      + '\n  Train Sensitivity: '
                                      + str(result[0][1])
                                      + '\n  Train Specificity: '
                                      + str(result[0][2])
                                      + '\nVALIDATION RESULTS'
                                      + '\n  Percentage of '
                                      + _DETECTION_PROBLEM
                                      + ' samples: '
                                      + str(percent_pos_train)
                                      + '\n  Validation acc: '
                                      + str(result[0][3])
                                      + '\n  Validation Sensitivity: '
                                      + str(result[0][4])
                                      + '\n  Validation Specificity: '
                                      + str(result[0][5]),
                                      file=out_f)
                            record_settings_results(model_name,
                                                    settings)
                            save_model(model, model_name)
                            mod_id += 1
                            
    if _LOAD_MODEL:
        prediction = {}
        for model_name in get_model_paths():
            model_name = 'model_9'
            warning_msg = ('Warning: ' + model_name
                           + ' has been discarded from the '
                           + 'evaluation process, due to -> ')
            [model, conf] = load_model(model_name)
            ##########
            conf['random_seed'] = _SEED
            conf['train_patient'] = train_patient
            conf['val_patient'] = val_patient
            conf['problem'] = _DETECTION_PROBLEM
            conf['n_train'] = 185984
            conf['n_validation'] = 6400
            conf['model_conf'] = ('C(32,11)-P-DR(0.25)-C(32,5)-P-'
                                  + 'DR(0.25)-C(64,3)-P-DR(0.25)-'
                                  + 'C(64,3)-P-DR(0.25)-DN(128)-'
                                  + 'D(0.25)-DN(1,Sigmoid)|'
                                  + 'INIT:glorot_normal|REGULAR:None|'
                                  + 'OPT:adadelta')
            conf['n_epoch'] = 100
            conf['n_parameter'] = 125473
            conf['percent_pos_val'] = 0.13
            conf['percent_pos_train'] = 0.14
            conf['augment_shift'] = _N_SHIFT
            conf['augment_rotate'] = _N_ROTATE
            conf['normalize'] = False
            conf['final_status'] = 'OK'
            ##########
            if conf['final_status'] == 'OK':
                if _REPRODUCIBILITY:
                    np.random.seed(conf['random_seed'])
                    rd.seed(conf['random_seed'])
                [status, predicted] = predict_result(
                    model, train_patient=conf['train_patient'],
                    validation_patient=conf['val_patient'],
                    problem=conf['problem'],
                    n_train=conf['n_train'],
                    n_val=conf['n_validation'],
                    threshold=conf['threshold'],
                    batch_size=conf['batch_size'],
                    augment_shift=conf['augment_shift'],
                    augment_rotate=conf['augment_rotate'],
                    normalize=conf['normalize'],
                    window_size=conf['window_size'])
                if status == 'OK':
                    prediction[model_name] = predicted
                else:
                    print(warning_msg
                          + 'its status produced while predicting.')
            else:
                print(warning_msg
                      + 'its status loaded from the conf file.')
            save_prediction_result(prediction)
            print('Fin')
            exit(1)
    print('END')

# EOF

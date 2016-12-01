"""Experiment configuration"""

# Authors: Julia Camps <julia.camps.sereix@est.fib.upc.edu>
# Created on: 30/11/2016 09:26

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
from FOG.metrics import record_metrics_result
from FOG.definitions import _get_prediction_global_key
from FOG.definitions import _get_prediction_partial_key
from FOG.experiment_conf import _get_rotate_augment
from FOG.experiment_conf import _get_shift_augment


def train_model(model, train_patient, problem, n_epoch, n_train,
                batch_size, window_size, validation_patient=None,
                n_val=0, filter_threshold=0.0):
    """Train model on the selected patients

    Parameters
    ----------
    model : keras.Sequential()
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
    train_file =
    [train_file, validation_file] = get_train_validation_path(
        train_patient, validation_patient, problem=problem)
    
    train_generator = generate_arrays_from_file(
        train_file, window_size, batch_size=batch_size,
        augment_count=[_get_shift_augment(), _get_rotate_augment()],
        filter_threshold=filter_threshold)
    
    if validation_file is not None:
        validation_generator = generate_arrays_from_file(
            validation_file, window_size, batch_size=batch_size,
            filter_threshold=filter_threshold)
    else:
        validation_generator = None
    
    [trained_model, result] = single_train(
        model, train_generator,
        validation_generator=validation_generator, n_epoch=n_epoch,
        n_train=n_train, n_val=n_val, batch_size=batch_size)
    
    return [trained_model, result]


def single_train(model, train_generator, validation_generator=None,
                 n_epoch=1, n_train=0,
                 n_val=0, batch_size=64):
    """Train the model


    """
    
    acc = 0
    time_count = time.clock()
    epochs = 0
    status = 'OK'
    statistics_record = []
    
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
                model.train_on_batch(X, y)
            except Exception as e:
                status = ('ERROR: While training the model: '
                          + str(repr(e)))
                end_epoch = True
                abort_train = True
            else:
                samples_epoch_count += batch_size
            
            if (samples_epoch_count + 1) >= n_train:
                end_epoch = True
            
            if end_epoch:
                # End loop condition (only condition)
                break
        if status == 'OK':
            [status, prediction] = predict_single_result(
                model, train_generator,
                validation_generator=validation_generator,
                n_train=n_train, n_val=n_val,
                batch_size=batch_size, verbose=True)
            if status != 'OK':
                abort_train = True
            else:
                statistics_record.append(prediction)
            epochs += 1
        print('==================\nENDED EPOCH: ' + str(epoch_it)
              + '\n========================')
    
    if not abort_train:
        print('TRAINING FINISHED WITH VALIDATION ACC: ' + str(acc))
    else:
        print('TRAINING WAS ABORTED')
    
    return [model, [epochs, (time.clock() - time_count), status,
                    statistics_record]]


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
    
    return predict_single_result(
        model, train_generator,
        validation_generator=validation_generator, n_train=n_train,
        n_val=n_val, batch_size=batch_size, verbose=True)


def predict_single_result(model, train_generator,
                          validation_generator=None, n_train=0,
                          n_val=0, batch_size=64, verbose=False):
    """"""
    result = {}
    [status, train_conf_mat, train_statistic] = \
        get_statistics(model, train_generator, n_train, batch_size,
                       msg='train_error_reproduction')
    if status == 'OK':
        if verbose:
            print('train prediction:')
            
        result['train'] = {
            _get_prediction_partial_key(): train_statistic,
            _get_prediction_global_key(): record_metrics_result(
                train_conf_mat, verbose=verbose)
        }
        
        if validation_generator is not None:
            [status, val_conf_mat, val_statistic] = \
                get_statistics(model, validation_generator,
                               n_val, batch_size,
                               msg=
                               'validation_error_reproduction')
            if status == 'OK':
                if verbose:
                    print('validation prediction:')
                    
                result['validation'] = {
                    _get_prediction_partial_key(): val_statistic,
                    _get_prediction_global_key():
                        record_metrics_result(val_conf_mat,
                                              verbose=verbose)
                    }
    
    return [status, result]



    
    
# EOF

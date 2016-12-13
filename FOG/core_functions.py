"""Experiment configuration"""

# Authors: Julia Camps <julia.camps.sereix@est.fib.upc.edu>
# Created on: 30/11/2016 09:26

import time
from collections import OrderedDict


from FOG.preprocessing_tools import get_generator
from FOG.metrics import get_statistics
from FOG.metrics import metrics_calc

from FOG.io_functions import report_event
from FOG.definitions import get_status_ini
from FOG.definitions import check_status
from FOG.definitions import get_prediction_partial_key
from FOG.definitions import get_prediction_global_key
from FOG.definitions import define_settings
from FOG.utils import to_string
from FOG.definitions import get_prediction_summary


def train_model(model, train_patient, n_epoch, n_train,
                batch_size, window_size, temporal_dependency, problem,
                validation_patient=None, n_validation=0, settings=None):
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

    train_generator, validation_generator, settings = get_generator(
        train_patient, window_size, batch_size,
        temporal_dependency, problem,
        validation_patient=validation_patient, settings=settings)

    [status, trained_model, train_summary, settings] = single_train(
        model, train_generator, n_epoch, n_train, batch_size,
        validation_generator=validation_generator,
        n_validation=n_validation,
        temporal=temporal_dependency, settings=settings)
    if check_status(status):
        msg = 'OK: Training process finished successfully'
        is_error = False
    else:
        msg = 'ERROR: Training process FAILED: ' + str(status)
        is_error = True
    report_event(msg, is_error=is_error)
    
    if settings is not None:
        settings = define_settings(settings, final_status=status)

    return [trained_model, train_summary, settings]


def single_train(model, train_generator, n_epoch, n_train,
                 batch_size, temporal, validation_generator=None,
                 n_validation=0, settings=None):
    """Train the model


    """
    summary = None
    time_ini = time.clock()
    status = get_status_ini()
    train_evolution = []
    is_error = False
    for epoch_it in range(n_epoch):
        msg = ('\n==================\nSTARTING EPOCH: '
               + str(epoch_it) + '\n========================')
        report_event(msg, is_run_log=True)
        samples_it = 0
        for [X, y, y_orig], additional_info in train_generator:
            current_patinet = additional_info['patient']
            current_file = additional_info['file']
            reset_state = additional_info['reset_state']
            if temporal and reset_state:
                model.reset_states()
            try:
                model.train_on_batch(X, y)
            except Exception as e:
                status = str(repr(e))
                msg = ('ERROR:  Train failed for epoch '
                         + str(epoch_it) + ' during patient: '
                         + current_patinet + ', and file: '
                         + current_file + ', and status: ' + status)
                is_error = True
            else:
                msg = ('OK: Train successful for epoch '
                         + str(epoch_it))
                is_error = False
                samples_it += batch_size
            # End loop only condition
            if (not check_status(status) or
                    (samples_it + batch_size) > n_train):
                break
            
        report_event(msg, is_error=is_error)
        report_event('==================\nENDED EPOCH: ' + str(
            epoch_it) + '\n========================',
                     is_run_log=True)
                
        if check_status(status):
            [status, prediction, summary] = predict_single_result(
                model, train_generator, batch_size,
                validation_generator=validation_generator,
                n_train=n_train, n_validation=n_validation)
            if check_status(status):
                train_evolution.append(['Epoch ' + str(epoch_it),
                                        prediction])
                msg = ('Prediction results:\n' + to_string(summary))
                is_run_log = True
                is_error = False
            else:
                msg = ('ERROR: Prediction failed at epoch '
                       + str(epoch_it) + ': ' + status)
                is_run_log = False
                is_error = True
            report_event(msg, is_run_log=is_run_log,
                         is_error=is_error)
            
    if check_status(status):
        msg = ('OK: Train finished successfully at epoch '
               + str(epoch_it))
        is_error = False
    else:
        msg = ('ERROR: Aborting train at epoch ' + str(epoch_it)
               + ', something has failed: ' + status)
        is_error = True
    report_event(msg, is_error=is_error)
    
    if settings is not None:
        settings = define_settings(settings, training_time=(
            time.clock() - time_ini), new_settings_dict=summary)
    
    return [status, model, train_evolution, settings]


def predict_result(model, train_patient, window_size, batch_size,
                   temporal, problem, n_train, threshold,
                   augment_shift, augment_rotate,
                   validation_patient=None, n_validation=None):
    """"""

    train_generator, validation_generator = get_generator(
        train_patient, window_size, batch_size, temporal,
        problem, validation_patient=validation_patient,
        shift_augmentation=augment_shift,
        rotate_augmentation=augment_rotate, threshold=threshold)
    
    return predict_single_result(
        model, train_generator,
        validation_generator=validation_generator, n_train=n_train,
        n_validation=n_validation, batch_size=batch_size)


def predict_label(model, generator, batch_size, n_train, temporal):
    """"""
    status = get_status_ini()
    label_data = []
    samples_it = 0
    for [X, y_true, y_orig], additional_info in generator:
        if temporal and additional_info['reset_state']:
            model.reset_states()
        try:
            y_pred = model.predict_on_batch(X)
        except Exception as e:
            status = str(repr(e))
        else:
            label_data.append((y_true, y_orig, y_pred))
            samples_it += batch_size
            # End loop only condition
            if (not check_status(status)
                    or (samples_it + batch_size) > n_train):
                break
    return status, label_data


def predict_single_result(model, train_generator, batch_size, n_train,
                          temporal, validation_generator=None,
                          n_validation=None):
    """"""
    train_metrics = None
    val_metrics = None
    prediction = OrderedDict([])
    status, label_data = predict_label(model, train_generator,
                                       batch_size, n_train, temporal)
    
    if check_status(status):
        [train_conf_mat, train_statistic] = get_statistics(label_data)
        msg = 'OK: Prediction successful for Train data'
        report_event(msg)
        train_metrics = metrics_calc(train_conf_mat)
        prediction['train'] = OrderedDict([
            (get_prediction_global_key(), train_metrics),
            (get_prediction_partial_key(), train_statistic)])

        if validation_generator is None or n_validation is None:
            report_event('WARNING: No validation generator or '
                         'counter')
        else:
            status, label_data = predict_label(
                model, validation_generator, batch_size, n_validation,
                temporal)
            
            if check_status(status):
                [val_conf_mat, val_statistic] = get_statistics(
                    label_data)
                msg = 'OK: Prediction successful for Validation data'
                report_event(msg)
                val_metrics = metrics_calc(val_conf_mat)
                prediction['validation'] = OrderedDict([
                    (get_prediction_global_key(), val_metrics),
                    (get_prediction_partial_key(), val_statistic)])
            else:
                msg = ('ERROR: Aborted prediction for Validation '
                       + 'data: ' + status)
                report_event(msg, is_error=True)
    else:
        msg = 'ERROR: Aborted prediction for Train data: ' + status
        report_event(msg, is_error=True)
    
    return [status, prediction, get_prediction_summary(
        train_metrics, val_metrics)]


# EOF


"""Experiment configuration"""

# Authors: Julia Camps <julia.camps.sereix@est.fib.upc.edu>
# Created on: 30/11/2016 09:26

import time
import numpy as np
from math import sqrt
# from os import mkdir
# from os import listdir
# import random as rd

from collections import OrderedDict

from FOG.io_functions import save_data
# from FOG.io_functions import prepare_data_path
# from FOG.io_functions import add_freq_to_data_path
# from FOG.io_functions import add_patient_to_data_path
# from FOG.io_functions import read_data_file

from FOG.preprocessing_tools import get_generator
# from FOG.preprocessing_tools import get_patient_split
#
# from FOG.metrics import get_statistics


# from FOG.io_functions import report_event
# from FOG.definitions import get_status_ini
# from FOG.definitions import check_status
# from FOG.definitions import get_prediction_partial_key
# from FOG.definitions import get_prediction_global_key
# from FOG.definitions import define_settings
# from FOG.definitions import get_data_dict
# from FOG.definitions import get_precalculated_data_path
# from FOG.definitions import get_patient_partition

# from FOG.definitions import get_prediction_summary

from FOG.utils import to_string

from keras.callbacks import LambdaCallback
# from keras.callbacks import History
# from keras.callbacks import Callback


# def generate_all_data(augmentation, window_time, batch_size, stacking,
#                       pure_threshold, temporal, n_feature,
#                       exclude_freq=[]):
#     """"""
#     # Get data dict and paths
#     data_dict = get_data_dict()
#     data_path = get_precalculated_data_path()
#     prepare_data_path(data_path)
#     train_patient, val_patient, test_patient = \
#         get_patient_partition()
#
#     # Generate augmentation
#     for data_freq, data in data_dict.items():
#         if data_freq in exclude_freq:
#             continue
#         add_freq_to_data_path(data_path, data_freq)
#         window_size = int(window_time * data_freq)
#         for patient, patient_files in data.items():
#             if patient in train_patient:
#                 patient_type = 'train'
#                 patient_augm = augmentation
#             else:
#                 continue
#             # elif patient in val_patient:
#             #     patient_type = 'val'
#             #     patient_augm = 1
#             # elif patient in test_patient:
#             #     patient_type = 'test'
#             #     patient_augm = 1
#             # else:
#             #     continue
#             patient_path = add_patient_to_data_path(
#                 data_path, patient, data_freq, patient_type)
#             generate_patient_data(patient_path, patient_files,
#                                   patient_augm, window_size,
#                                   batch_size, stacking,
#                                   pure_threshold, data_freq, temporal)


# def generate_patient_data(patient_path, patient_files,
#                           patient_augm, window_size, batch_size,
#                           stacking, pure_threshold, data_freq,
#                           temporal, n_feature):
#     """"""
#     roate_proba = max(0., min(0.5, (patient_augm - 4) * 0.1))
#     for i in range(patient_augm):
#         new_file_path = patient_path + '/' + str(i)
#         mkdir(new_file_path)
#         predict = (i == 1)
#         for data_file in patient_files:
#             patient_data = read_data_file(data_file)
#             file_generator, aux_val, aux_settings = \
#                 get_generator(
#                 patient_data, window_size, batch_size, stacking,
#                 pure_threshold, data_freq, n_feature,
#                     validation_data=None,
#                 settings=None, temporal=temporal, predict=predict,
#                 get_raw_data=False, single_file=True, roate_proba=roate_proba)
#             generate_file_transformed(file_generator,
#                                       new_file_path, data_file,
#                                       batch_size, window_size)
#
#
# def generate_file_transformed(file_generator, new_file_path,
#                               data_file, batch_size, window_size):
#     """"""
#     X_total = None
#     y_true_total = np.array([])
#     for X_batch, y_batch in file_generator:
#         for it in range(batch_size):
#             X = X_batch[it]
#             y_true = y_batch[it]
#             y_true_total = np.concatenate(
#                 (y_true_total,
#                  np.full(window_size, y_true, dtype=int)), axis=0)
#
#             if X_total is None:
#                 X_total = X
#             else:
#                 X_total = np.concatenate((X_total, X))
#
#     complete_data = np.concatenate((X_total,
#                                     y_true_total[:, np.newaxis]),
#                                    axis=1)
#     save_data(data=complete_data,
#               file_path=new_file_path+'/'+data_file[(data_file.rfind(
#                   '/') + 1):])
#
#
# def load_precalculated(data_freq, batch_size, window_size, temporal):
#     """"""
#     train_data_path = get_precalculated_data_path(
#         data_freq=data_freq, data_type='train')
#     # val_data_path = get_precalculated_data_path(
#     #     data_freq=data_freq, data_type='val')
#     train_data = _load_precalculated(train_data_path)
#     # val_data = _load_precalculated(val_data_path)
#     train_data = represent_data(train_data, batch_size,
#                                 window_size, temporal)
#     # val_data = represent_data(val_data, batch_size, window_size,
#     #                           temporal)
#
#     return train_data  #, val_data
#
#
# def _load_precalculated(data_path):
#     """"""
#     files_dict = defaultdict(list)
#     for patient in listdir(data_path):
#         patient_path = data_path + '/' + patient
#         for augment in listdir(patient_path):
#             augment_path = patient_path + '/' + augment
#             for file in listdir(augment_path):
#                 file_path = augment_path + '/' + file
#                 file_name = file[:file.rfind('.')]
#                 files_dict[file_name].append(read_data_file(file_path))
#     return files_dict


# def represent_data(data, batch_size, window_size, temporal):
#     """"""
#     if temporal:
#         data_struct = []
#     else:
#         data_struct = defaultdict(list)
#     for file_name, files in data.items():
#         for file in files:
#             file_data = []
#             batch_it = 0
#             batch_X = []
#             batch_Y = []
#             for i in range(0, len(file), window_size):
#                 window = file[i:i+window_size]
#                 X = window[:,:-1]
#                 Y = list(set(window[:,-1]))[0]
#                 if temporal:
#                     batch_X.append(X)
#                     batch_Y.append(Y)
#                     batch_it += 1
#                     if batch_it == batch_size:
#                         file_data.append((np.asarray(batch_X),
#                                           np.asarray(batch_Y)))
#                         batch_it = 0
#                         batch_X = []
#                         batch_Y = []
#                 else:
#                     data_struct['x'].append(np.asarray(X))
#                     data_struct['y'].append(np.asarray(Y))
#             if temporal:
#                 data_struct.append(file_data)
#     return data_struct


def calc_metrics(conf_mat):
    """"""
    accuracy = ((conf_mat[0, 0] + conf_mat[1, 1])
                / sum(sum(conf_mat)))
    precision = (conf_mat[0, 0]
                         / (conf_mat[0, 0] + conf_mat[1, 0]))
    recall = (conf_mat[0, 0]
                         / (conf_mat[0, 0] + conf_mat[0, 1]))
    specificity = (conf_mat[1, 1]
              / (conf_mat[1, 1] + conf_mat[1, 0]))
    return accuracy, precision, recall, specificity
    

def conf_to_string(model_name, configuration):
    """"""
    relevant_conf = [
        ('model_name', model_name),
        ('stacking', configuration['stacking']),
        ('dropout', configuration['dropout']),
        ('batch_size', configuration['batch_size']),
        ('window_time', configuration['window_time']),
        ('learning_rate', configuration['learning_rate']),
        ('weight_init', configuration['weight_init']),
        ('optimizer', configuration['optimizer']),
        ('objective', configuration['objective']),
        ('penalty', configuration['penalty']),
        ('regularization', configuration['regularization']),
        ('regularization_value',
         configuration['regularization_value']),
        ('window_size', configuration['window_size']),
        ('temporal', configuration['temporal']),
        ('data_freq', configuration['data_freq']),
        ('pure_threshold', configuration['pure_threshold']),
        ('n_epoch', configuration['n_epoch']),
        ('activation_last_layer',
         configuration['activation_last_layer']),
        ('regularization_last_layer',
         configuration['regularization_last_layer']),
        ('conv_kernel', configuration['conv_kernel']),
        ('conv_width', configuration['conv_width']),
        ('dense_width', configuration['dense_width']),
        ('conv_layers', configuration['conv_layers']),
        ('dense_layers', configuration['dense_layers']),
        ('n_feature', configuration['n_feature']),
        ('augmentation', configuration['augmentation']),
        ('lstm_dropout', configuration['lstm_dropout']),
        ('roate_proba', configuration['roate_proba'])
    ]
    just_conf = [str(item[1]) for item in relevant_conf]
    new_conf = ' '.join(just_conf)+'\n'
    return new_conf


def add_configuration(model_name, configuration):
    """"""
    with open('prediction/model_confs.csv', 'a') as confs:
        confs.write(conf_to_string(model_name, configuration))


def predict_model(model, data, batch_size, window_size, temporal,
                  stacking, pure_threshold, data_freq,
                  model_name, reduce_memory, n_feature):
    """"""
    for data_name, data_files_generator in data.items():
        for patient_name, patient_file, patient_data in \
                data_files_generator:
            file_data_generator, aux_val, aux_settings = \
                get_generator(
                patient_data, window_size, batch_size, stacking,
                pure_threshold, data_freq, n_feature,
                    validation_data=None,
                settings=None, temporal=temporal, predict=True,
                get_raw_data=True, single_file=True)

            X_total = None
            X_raw_total = None
            y_true_total = np.array([])
            y_pred_total = np.array([])
            for [X_spe_batch, X_tem_batch, pre_X_tem_batch], \
                y_true_batch in \
                    file_data_generator:
                y_pred_batch = np.sign(model.predict_on_batch([
                    X_spe_batch, X_tem_batch, pre_X_tem_batch]))
                for it in range(batch_size):
                    X_spe = X_spe_batch[it]
                    y_true = y_true_batch[it]
                    y_pred = y_pred_batch[it]
                    X_tem = X_tem_batch[it]
                    y_true_total = np.concatenate(
                        (y_true_total,
                         np.full(window_size, y_true, dtype=int)), axis=0)
        
                    y_pred_total = np.concatenate((y_pred_total, np.full(
                        window_size, y_pred, dtype=int)), axis=0)
                    if X_total is None:
                        X_total = X_spe
                        X_raw_total = X_tem
                    else:
                        X_total = np.concatenate((X_total, X_spe))
                        X_raw_total = np.concatenate((X_raw_total, X_tem))
            if temporal:
                model.reset_states()
            if X_total is not None:
                # print(X_total.shape)
                # print(X_raw_total.shape)
                # print(y_true_total.shape)
                # print(y_pred_total.shape)
                complete_data = np.concatenate(
                    (X_raw_total[:, 3:],
                     y_true_total[:, np.newaxis],
                     y_pred_total[:, np.newaxis]), axis=1)
                
    
                save_data(data=complete_data,
                          file_path='prediction/'+data_name
                                    +'/'+ model_name+patient_file[
                                      (patient_file.rfind('/')+1):],
                          replace=True)


# def evaluate_epochs(model, n_epoch, train_data, val_data,
#                     batch_size, n_train, n_val, window_size,
#                     stacking, pure_threshold, data_freq, temporal, n_feature):
#     """"""
#     cum_sensitivity = 0.
#     cum_specificity = 0.
#     train_generator, val_generator, aux = get_generator(
#         train_data, window_size, batch_size,
#         stacking, pure_threshold, data_freq, n_feature,
#         validation_data=val_data, temporal=temporal)
#     for epoch_i in range(n_epoch):
#         samples_it = 0
#         conf_mat = np.zeros((2, 2))
#         model.fit_generator(train_generator, n_train, 1,
#                             max_q_size=20,
#                             nb_worker=1, pickle_safe=False)
#         for X, y_true in val_generator:
#             y_pred = model.predict_on_batch(X)
#             y_pred = np.sign(y_pred)
#             positives = np.maximum(y_true, 0.)
#             pred_pos = np.maximum(y_pred, 0.)
#             negatives = np.maximum(-y_true, 0.)
#             pred_neg = np.maximum(-y_pred, 0.)
#             true_pos = pred_pos * positives
#             false_pos = pred_pos * negatives
#             true_neg = pred_neg * negatives
#             false_neg = pred_neg * positives
#             conf_mat[0, 0] += np.sum(true_pos)
#             conf_mat[0, 1] += np.sum(false_neg)
#             conf_mat[1, 0] += np.sum(false_pos)
#             conf_mat[1, 1] += np.sum(true_neg)
#
#             samples_it += batch_size
#             # End loop only condition
#             if (samples_it + batch_size) > n_val:
#                 break
#
#         accuracy, sensitivity, specificity = calc_metrics(conf_mat)
#
#         cum_specificity += specificity
#         cum_sensitivity += sensitivity
#
#     # AVERAGE METRICS
#     avg_sen = cum_sensitivity / n_epoch
#     avg_spe = cum_specificity / n_epoch
#     g_avg_sen_spe = sqrt(avg_sen*avg_spe)
#
#     return model, g_avg_sen_spe


def train_model(model, train_data, n_epoch, n_train,
                batch_size, window_size, stacking,
                pure_threshold, data_freq, augmentation_factor,
                n_batch_per_file, n_feature, roate_proba,
                validation_data=None,
                n_validation=0, settings=None, log_file_name=None,
                model_name='', summarize=False, summary_epochs=10,
                initial_epochs=0, epoch_step_size=40,
                metric_threshold=0.005, temporal=False,
                precalculated=False, train_precalculated_data=None,
                per_patient=True, save_train_error=True, conf_str=''):
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

    with open(log_file_name, 'a') as log_file:
        log_file.write(conf_str)

    success = True

    # summary = OrderedDict([])
    best_model = None
    patience_max = 1000
    patience_count = 0
    previous_metric = 0.
    metric_threshold = 0.005
    best_epoch_it = initial_epochs
    epoch_ini = 0
    epoch_step = 1
    best_metric = 0.
    abort_train = False

    # if temporal:
    #     sequence_length = n_batch_per_file
    #
    #     def reset_states(batch, logs):
    #         if batch % sequence_length == 0:
    #             model.reset_states()
    #
    #     callbacks = [
    #         LambdaCallback(on_batch_end=reset_states)
    #     ]
    # else:
    #     callbacks = []
    # if initial_epochs > 0:
    #     train_generator, aux_val, settings = get_generator(
    #         train_data, window_size, batch_size,
    #         stacking, pure_threshold, data_freq, n_feature,
    #         augmentation_factor=augmentation_factor,
    #         validation_data=None,
    #         settings=settings, temporal=temporal, roate_proba=roate_proba)
    #
    #     model.fit_generator(train_generator,
    #                         int(n_train*augmentation_factor/batch_size),
    #                     initial_epochs,
    #                     max_q_size=40,
    #                     callbacks=callbacks,
    #                     nb_worker=1, pickle_safe=False, verbose=0)

    for epoch_i in range(epoch_ini, n_epoch, epoch_step):
        ini_time = time.monotonic()
        if abort_train:
            continue

        train_generator, aux_val, aux_settings = \
            get_generator(
                train_data, window_size, batch_size,
                stacking, pure_threshold, data_freq,
                n_feature,
                validation_data=None,
                settings=settings, temporal=temporal,
                augmentation_factor=augmentation_factor,
                roate_proba=roate_proba)

        batch_it = 0
        for [X_spe, X_tem, X_pre_tem], y_true in train_generator:
            model.train_on_batch([X_spe, X_tem, X_pre_tem], y=y_true)
            batch_it += 1
            if batch_it % n_batch_per_file == 0:
                model.reset_states()
            if batch_it > int(n_train * augmentation_factor /
                                      batch_size):
                break
        # history = model.fit_generator(
        # train_generator, steps_per_epoch=int(n_train * augmentation_factor /
        #                      batch_size), callbacks=None,
        #     epochs=epoch_step, max_q_size=10, workers=1,
        #     pickle_safe=False, verbose=0)
        # summary[epoch_i] = OrderedDict([])

        train_acc = []
        train_sen = []
        train_spe = []
        train_pre = []
        for patient_name, patient_data in train_data.items():
            conf_mat = np.zeros((2, 2))
            for patient_file in patient_data:
                train_eval_generator, aux_val, aux_settings = \
                    get_generator(
                        patient_file, window_size, batch_size,
                        stacking, pure_threshold,
                        data_freq, n_feature,
                        validation_data=None,
                        settings=settings, temporal=temporal,
                        predict=True, single_file=True)

                for [X_spe, X_tem, X_pre_tem], y_true in train_eval_generator:
                    y_pred = model.predict_on_batch([X_spe, X_tem, X_pre_tem])
                    y_pred = np.sign(y_pred).T[0]
                    positives = np.maximum(y_true, 0.)
                    pred_pos = np.maximum(y_pred, 0.)
                    negatives = np.maximum(-y_true, 0.)
                    pred_neg = np.maximum(-y_pred, 0.)
                    true_pos = pred_pos * positives
                    false_pos = pred_pos * negatives
                    true_neg = pred_neg * negatives
                    false_neg = pred_neg * positives
                    conf_mat[0, 0] += np.sum(true_pos)
                    conf_mat[0, 1] += np.sum(false_neg)
                    conf_mat[1, 0] += np.sum(false_pos)
                    conf_mat[1, 1] += np.sum(true_neg)

                if temporal:
                    model.reset_states()

            train_accuracy, train_precision, train_sensitivity, \
            train_specificity = \
                calc_metrics(conf_mat)

            train_acc.append(train_accuracy)
            train_sen.append(train_sensitivity)
            train_spe.append(train_specificity)
            train_pre.append(train_precision)

        train_accuracy = np.nanmean(np.asarray(train_acc))
        train_sensitivity = np.nanmean(np.asarray(train_sen))
        train_specificity = np.nanmean(np.asarray(train_spe))
        train_precision = np.nanmean(np.asarray(train_pre))

        # train_metric = sqrt(train_sensitivity * train_specificity)
        train_metric = 2 * (train_precision * train_sensitivity / (
            train_precision + train_sensitivity))

        # VAL METRICS
        val_acc = []
        val_sen = []
        val_spe = []
        val_pre = []
        for patient_name, patient_data in validation_data.items():
            conf_mat = np.zeros((2, 2))
            for patient_file in patient_data:
                val_eval_generator, aux_val, aux_settings = \
                    get_generator(
                        patient_file, window_size, batch_size,
                        stacking, pure_threshold,
                        data_freq, n_feature,
                        validation_data=None,
                        settings=settings, temporal=temporal,
                        predict=True, single_file=True)

                for [X_spe, X_tem, X_pre_tem], y_true in val_eval_generator:
                    y_pred = model.predict_on_batch([X_spe, X_tem, X_pre_tem])
                    y_pred = np.sign(y_pred).T[0]
                    positives = np.maximum(y_true, 0.)
                    pred_pos = np.maximum(y_pred, 0.)
                    negatives = np.maximum(-y_true, 0.)
                    pred_neg = np.maximum(-y_pred, 0.)
                    true_pos = pred_pos * positives
                    false_pos = pred_pos * negatives
                    true_neg = pred_neg * negatives
                    false_neg = pred_neg * positives
                    conf_mat[0, 0] += np.sum(true_pos)
                    conf_mat[0, 1] += np.sum(false_neg)
                    conf_mat[1, 0] += np.sum(false_pos)
                    conf_mat[1, 1] += np.sum(true_neg)

                if temporal:
                    model.reset_states()

            val_accuracy, val_precision, val_sensitivity, \
            val_specificity = \
                calc_metrics(conf_mat)

            val_acc.append(val_accuracy)
            val_sen.append(val_sensitivity)
            val_spe.append(val_specificity)
            val_pre.append(val_pre)

        val_accuracy = np.nanmean(np.asarray(val_acc))
        val_sensitivity = np.nanmean(np.asarray(val_sen))
        val_specificity = np.nanmean(np.asarray(val_spe))
        val_precision = np.nanmean(np.asarray(val_spe))

        # val_metric = sqrt(val_sensitivity*val_specificity)
        val_metric = 2*(val_precision * val_sensitivity/(
            val_precision + val_sensitivity))

        new_metric = min(train_metric, val_metric)

        patience_count += 1
        if new_metric > best_metric:
            # if best_metric < 0.8:
            best_metric = new_metric
            best_model = model.get_weights()
            best_epoch_it = epoch_i + initial_epochs
            if new_metric > (previous_metric +
                                 metric_threshold):
                previous_metric = new_metric
                patience_count = 0
            # elif min(train_specificity, val_specificity) > 0.9:
            #     best_metric = new_metric
            #     best_model = model.get_weights()
            #     best_epoch_it = epoch_i + initial_epochs
            #     if new_metric > (previous_metric +
            #                          metric_threshold):
            #         previous_metric = new_metric
            #         patience_count = 0

        # if epoch_i > 30 and train_metric < 0.1:
        #     abort_train = True
        #     success = False
        # elif epoch_i > 50 and train_metric < 0.5:
        #     abort_train = True
        #     success = False
        # elif epoch_i > 100 and best_metric < 0.6:
        #     abort_train = True
        #     success = False
        # elif epoch_i > 200 and best_metric < 0.7:
        #     abort_train = True
        #     success = False
        # elif epoch_i > 300 and best_metric < 0.8:
        #     abort_train = True
        #     success = False
        # el
        if patience_count > patience_max:
            abort_train = True
            print('STOPPED BY CONVERGENCE')

        # PRINT METRICS
        if temporal==2:
            prefix = 'GRU2 - '
        elif temporal == 1:
            prefix = 'LSTM2 - '
        elif temporal == 0:
            prefix = 'MLP2 - '
        elif temporal == 3:
            prefix = 'GRU1-MLP1 - '
        elif temporal == 4:
            prefix = 'YEAH - '
        print('\n' + prefix + model_name + ' - Ep: ' + str(epoch_i+initial_epochs)
              + ' - patience_count: ' + str(patience_count) + '/' + str(patience_max)
              + ' -  time: ' + str(time.monotonic() - ini_time)
              # + ' -  loss: ' + str(history.history['loss'][0])
              + '\ntrain_precision: ' + str(round(train_precision, 4)*100)
              + ' - train_recall: ' + str(round(train_sensitivity, 4)*100)
              + ' - train_f1: ' + str(round(train_metric, 4) * 100)
              + ' - val_precision: ' + str(round(val_precision,4)*100)
              + ' - val_recall: ' + str(round(val_sensitivity,4)*100)
              + ' - val_f1: ' + str(round(val_metric, 4)*100)
              + ' - best_f1: ' + str(round(best_metric, 4) * 100)
              + ' - train_acc: ' + str(round(train_accuracy, 4) * 100)
              + ' - val_acc: ' + str(
            round(val_accuracy, 4) * 100)
              + ' - train_spe: ' + str(round(train_specificity, 4) * 100)
              + ' - val_spe: ' + str(
            round(val_specificity, 4) * 100)
              )
        train_aux = [[epoch_i,
                      train_accuracy,
                      train_precision,
                      train_sensitivity,
                      train_specificity,
                      val_accuracy, val_precision, val_sensitivity,
                      val_specificity, train_metric, val_metric, new_metric
                      ]]

        with open(log_file_name, 'a') as log_file:
            log_file.write(to_string(train_aux)+'\n')
            
    if best_model is not None:
        model.set_weights(best_model)

    return [model, best_epoch_it, success]


# EOF


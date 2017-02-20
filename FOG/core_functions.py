"""Experiment configuration"""

# Authors: Julia Camps <julia.camps.sereix@est.fib.upc.edu>
# Created on: 30/11/2016 09:26

import time
import numpy as np
from math import sqrt

from collections import OrderedDict

# from keras.callbacks import CSVLogger
# from keras.callbacks import LambdaCallback

from FOG.io_functions import save_data

from FOG.preprocessing_tools import get_generator
from FOG.preprocessing_tools import get_prediction_generator

from FOG.metrics import get_statistics
# from FOG.metrics import metrics_calc

from FOG.io_functions import report_event
from FOG.definitions import get_status_ini
from FOG.definitions import check_status
from FOG.definitions import get_prediction_partial_key
from FOG.definitions import get_prediction_global_key
from FOG.definitions import define_settings
# from FOG.definitions import label_is_positive
from FOG.definitions import get_prediction_summary

from FOG.utils import to_string

# import matplotlib.pyplot as plt

#
# from matplotlib import pyplot as plt
#
#
#
# train_summary = OrderedDict([])

# epoch_conf_mat = np.zeros((2, 2))
# epoch_y_pred = np.array(([],))
# epoch_y_true = np.array(([],))
# epoch_true_pos = 0.
# epoch_true_neg = 0.
# epoch_positives = 0.
# epoch_negatives = 0.

#
# epoch_it = 0


# def save_results(logs):
#     """"""
#     # global epoch_y_pred
#     # global epoch_y_true
#     # if epoch_y_pred is None:
#     #     epoch_y_pred = np.sign(logs['y_pred'])
#     #     epoch_y_true = logs['y_true']
#     # else:
#     #     epoch_y_pred = np.concatenate(
#     #         (epoch_y_pred, np.sign(logs['y_pred'])))
#     #     epoch_y_true = np.concatenate((epoch_y_true, logs['y_true']))
#     global epoch_true_pos
#     global epoch_true_neg
#     global epoch_positives
#     global epoch_negatives
#     epoch_true_pos += logs['TP']
#     epoch_true_neg += logs['TN']
#     epoch_positives += logs['P']
#     epoch_negatives += logs['N']
        
    # print((np.sign(logs['y_pred'])).shape)
    # print(epoch_y_pred.shape)
    # print((np.sign(logs['y_pred'])).ndim)
    # print(epoch_y_pred.ndim)
    
    # y_pred = np.sign(logs['y_pred'])
    # y_true = logs['y_true']
    # positives = np.maximum(y_true, 0.)
    # pred_pos = np.maximum(y_pred, 0.)
    # negatives = np.maximum(-y_true, 0.)
    # pred_neg = np.maximum(-y_pred, 0.)
    # true_pos = pred_pos * positives
    # false_pos = pred_pos - true_pos
    # true_neg = pred_neg * negatives
    # false_neg = pred_neg - true_neg
    # epoch_conf_mat[0, 0] += np.sum(true_pos)
    # epoch_conf_mat[0, 1] += np.sum(false_neg)
    # epoch_conf_mat[1, 0] += np.sum(false_pos)
    # epoch_conf_mat[1, 1] += np.sum(true_neg)

#
# def save_epoch_result(epoch, logs):
#     """"""
#
#     # global train_conf_mat
#     # train_conf_mat[epoch] = metrics_calc()
#     # y_true = logs['results']


def calc_metrics(conf_mat):
    """"""
    accuracy = ((conf_mat[0, 0] + conf_mat[1, 1])
                / sum(sum(conf_mat)))
    sensitivity = (conf_mat[0, 0]
                         / (conf_mat[0, 0] + conf_mat[0, 1]))
    specificity = (conf_mat[1, 1]
                   / (conf_mat[1, 0] + conf_mat[1, 1]))
    return accuracy, sensitivity, specificity
    

def add_configuration(model_name, configuration):
    """"""
    relevant_conf = [
        ('model_name', model_name),
        ('cutting', configuration['cutting']),
        ('dropout', configuration['dropout']),
        ('batch_size', configuration['batch_size']),
        ('window_time', configuration['window_time']),
        ('learning_rate', configuration['learning_rate']),
        ('weight_init', configuration['weight_init']),
        ('optimizer', configuration['optimizer']),
        ('penalty', configuration['penalty']),
        ('regularization', configuration['regularization']),
        ('regularization_value',
         configuration['regularization_value']),
        ('window_size', configuration['window_size']),
        ('temporal', configuration['temporal']),
        ('data_freq', configuration['data_freq']),
        ('shift', configuration['shift']),
        ('rotate', configuration['rotate']),
        ('pure_threshold', configuration['pure_threshold']),
        ('pos_threshold', configuration['pos_threshold']),
        ('percent_throw_no_fog',
         configuration['percent_throw_no_fog']),
        ('n_epoch', configuration['n_epoch']),
        ('activation_last_layer',
         configuration['activation_last_layer']),
        ('regularization_last_layer',
         configuration['regularization_last_layer']),
        ('conv_kernel', configuration['conv_kernel']),
        ('conv_width', configuration['conv_width']),
        ('dense_width', configuration['dense_width']),
        ('conv_layers', configuration['conv_layers']),
        ('dense_layers', configuration['dense_layers'])
    ]
    just_conf = [str(item[1]) for item in relevant_conf]
    new_conf = ' '.join(just_conf)+'\n'
    with open('prediction/model_confs.dat', 'a') as confs:
        confs.write(new_conf)


def predict_model(model, data, batch_size, window_size, temporal,
                  percent_throw_no_fog, cutting, pure_threshold,
                  problem, model_name, reduce_memory):
    """"""
    
    for data_name, data_files_generator in data.items():
        for patient_name, patient_file, patient_data in \
                data_files_generator:
            file_data_generator = get_prediction_generator(patient_data, window_size, batch_size,
                             cutting, pure_threshold,
                             percent_throw_no_fog, problem)

            # samples_it = 0
            X_total = None
            X_raw_total = None
            Y_raw_total = None
            y_true_total = np.array([])
            y_pred_total = np.array([])
            for X_batch, y_true_batch, X_raw_batch, Y_raw_batch in \
                    file_data_generator:
                y_pred_batch = np.sign(model.predict_on_batch(X_batch))
                for it in range(batch_size):
                    X = X_batch[it]
                    y_true = y_true_batch[it]
                    y_pred = y_pred_batch[it]
                    X_raw = X_raw_batch[it]
                    Y_raw = Y_raw_batch[it]
                    y_true_total = np.concatenate(
                        (y_true_total,
                         np.full(window_size, y_true, dtype=int)), axis=0)
        
                    y_pred_total = np.concatenate((y_pred_total, np.full(
                        window_size, y_pred, dtype=int)), axis=0)
                    if X_total is None:
                        X_total = X
                        X_raw_total = X_raw
                        Y_raw_total = Y_raw
                    else:
                        X_total = np.concatenate((X_total, X))
                        X_raw_total = np.concatenate((X_raw_total, X_raw))
                        Y_raw_total = np.concatenate(
                            (Y_raw_total, Y_raw))
        
                # samples_it += batch_size
                # End loop only condition
                # if (samples_it + batch_size) > n_validation:
                #     break

            # #ACC
            # # Plot transformation
            # ax = X_total[:, 3]
            # ay = X_total[:, 4]
            # az = X_total[:, 5]
        
            time_data = np.linspace(0, (
            X_total.shape[0] * (1 / 100)), X_total.shape[0])
            
            # print(time_data.shape)
            # print(X_raw_total.shape)
            # print(Y_raw_total.shape)
            # print(X_total.shape)
            # print(y_true_total.shape)
            # print(y_pred_total.shape)
            # y_concat = np.concatenate((y_true_total[:,np.newaxis],
            #                            y_pred_total[:,np.newaxis]),
            #                           axis=1)
            if reduce_memory:
                complete_data = np.concatenate(
                    (X_raw_total[:, 3:6],
                     Y_raw_total[:, np.newaxis],
                     y_true_total[:, np.newaxis],
                     y_pred_total[:, np.newaxis]), axis=1)
            else:
                complete_data = np.concatenate((time_data[:,np.newaxis], X_raw_total,
                                                Y_raw_total[:,np.newaxis],
                                                X_total, y_true_total[:,np.newaxis],
                                                y_pred_total[:,np.newaxis]), axis=1)

            save_data(data=complete_data,
                      file_path='prediction/'+data_name
                                +'/'+ model_name+patient_file[
                                  (patient_file.rfind('/')+1):])

    # # SPLIT VIEW
    # f1, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True,
    #                                              sharey=True)
    # ax1.set_title('ACC-CHANNELS')
    # ax1.set_xlabel('window-instances')
    #
    # ax1.plot(time_data, ax, color='r', label='accX')
    # ax2.plot(time_data, ay, color='g', label='accY')
    # ax3.plot(time_data, az, color='b', label='accZ')
    # ax4.plot(time_data, y_true_total, color='c',
    #          label='y_true')
    # ax5.plot(time_data, y_pred_total, color='m',
    #          label='y_pred')
    #
    # # SUMMARY VIEW
    # f2 = plt.figure()
    # ax1 = f2.add_subplot(111)
    #
    # ax1.set_title('ACC-SUMMARY')
    # ax1.set_xlabel('window-instances')
    #
    # ax1.plot(time_data, ax, color='r', label='accX')
    # ax1.plot(time_data, ay, color='g', label='accY')
    # ax1.plot(time_data, az, color='b', label='accZ')
    # ax1.plot(time_data, y_true_total, color='c',
    #          label='y_true')
    # ax1.plot(time_data, y_pred_total, color='m',
    #          label='y_pred')
    #
    # ax1.legend()
    #
    # plt.show()


def evaluate_epochs(model, n_epoch, generator, batch_size, n_data):
    """"""
    cum_sensitivity = 0.
    cum_specificity = 0.
    for epoch_i in range(n_epoch):
        samples_it = 0
        conf_mat = np.zeros((2, 2))
        for X, y_true in generator:
            y_pred = model.predict_on_batch(X)
            y_pred = np.sign(y_pred)
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
        
            samples_it += batch_size
            # End loop only condition
            if (samples_it + batch_size) > n_data:
                break

        accuracy, sensitivity, specificity = calc_metrics(conf_mat)

        cum_specificity += specificity
        cum_sensitivity += sensitivity

    # AVERAGE METRICS
    avg_sen = cum_sensitivity / n_epoch
    avg_spe = cum_specificity / n_epoch
    g_avg_sen_spe = sqrt(avg_sen*avg_spe)
    
    return model, g_avg_sen_spe
    
def train_model(model, train_data, n_epoch, n_train,
                batch_size, window_size, temporal, problem,
                percent_throw_no_fog, shift, rotate, pos_threshold, cutting,
                pure_threshold,
                validation_data=None,
                n_validation=0, settings=None, log_file_name=None,
                model_name='', summarize=True, summary_epochs=5,
                initial_epochs=20, epoch_step_size=10,
                metric_threshold=0.005):
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
    
    train_time = time.clock()
    # train_accuracy = -1
    # train_sensitivity = -1
    # train_specificity = -1
    # val_accuracy = -1
    # val_sensitivity = -1
    # val_specificity = -1
    train_generator, validation_generator, settings = get_generator(
        train_data, window_size, batch_size,
        temporal, problem, shift, rotate, cutting, pure_threshold,
        validation_data=validation_data,
        settings=settings, percent_throw_no_fog=percent_throw_no_fog, pos_threshold=pos_threshold)
    
    train_eval_generator, val_eval_generator, aux = get_generator(
        train_data, window_size, batch_size,
        temporal, problem, shift, rotate, cutting, pure_threshold,
        validation_data=validation_data,
        settings=settings, percent_throw_no_fog=percent_throw_no_fog,
        pos_threshold=pos_threshold)
    
    summary = OrderedDict([])
    model.fit_generator(train_generator, n_train,
                        initial_epochs,
                        max_q_size=20,
                        nb_worker=1, pickle_safe=False)
    epoch_count = initial_epochs
    n_epoch = n_epoch - initial_epochs
    # cum_train_specificity = 0.
    # cum_train_sensitivity = 0.
    # cum_val_specificity = 0.
    # cum_val_sensitivity = 0.
    train_finished = False
    model_weights_backup = model.get_weights()
    if summarize and n_epoch > 0:
        previous_metric = 0.
        for epoch_i in range(0, n_epoch, (epoch_step_size+summary_epochs)):
            if train_finished:
                continue
            
            model.fit_generator(train_generator, n_train, epoch_step_size,
                            # verbose=1,
                                # callbacks=[
                # save_batch_results_callback,
                # save_epoch_results_callback,
                # CSVLogger(log_file_name, separator=' ',
                #           append=True)
                # ],
                #             validation_data=validation_generator,
                #             nb_val_samples=n_validation,
                            # class_weight=class_weight,
                            max_q_size=20,
                            nb_worker=1, pickle_safe=False)
            model, new_metric = evaluate_epochs(
                model, summary_epochs, val_eval_generator, batch_size, n_validation)
            print('NEW METRIC VALUE: ')
            print(new_metric)
            if new_metric > (previous_metric + metric_threshold):
                previous_metric = new_metric
                model_weights_backup = model.get_weights()
                epoch_count += (epoch_step_size + summary_epochs)
                print('epoch_count: ')
                print(epoch_count)
            else:
                model.set_weights(model_weights_backup)
                train_finished = True
                
        cum_train_specificity = 0.
        cum_train_sensitivity = 0.
        cum_val_specificity = 0.
        cum_val_sensitivity = 0.
        for epoch_i in range(summary_epochs):
            # global epoch_y_pred
            # global epoch_y_true
            # epoch_y_pred = None
            # epoch_y_true = None
    
            # global epoch_true_pos
            # global epoch_true_neg
            # global epoch_positives
            # global epoch_negatives
            # epoch_true_pos = 0.
            # epoch_true_neg = 0.
            # epoch_positives = 0.
            # epoch_negatives = 0.
            # global epoch_conf_mat
            # epoch_conf_mat = np.zeros((2, 2))
            # print('Epoch ' + str(epoch_i) + '/' + str(n_epoch))
            # model.fit_generator(train_generator, n_train, 1,
            #                 verbose=1, callbacks=[
            #     # save_batch_results_callback,
            #     # save_epoch_results_callback,
            #     CSVLogger(log_file_name, separator=' ',
            #               append=True)
            #     ],
            #                 validation_data=validation_generator,
            #                 nb_val_samples=n_validation,
            #                 # class_weight=class_weight,
            #                 max_q_size=30,
            #                 nb_worker=1, pickle_safe=False)
            # epoch_train_time = time.clock() - epoch_time_ini
            # train_time += epoch_train_time
    
            # conf_mat = np.zeros((2, 2))
            # positives = np.maximum(epoch_y_true, 0.)
            # pred_pos = np.maximum(epoch_y_pred, 0.)
            # negatives = np.maximum(-epoch_y_true, 0.)
            # pred_neg = np.maximum(-epoch_y_pred, 0.)
            # true_pos = pred_pos * positives
            # false_pos = pred_pos - true_pos
            # true_neg = pred_neg * negatives
            # false_neg = pred_neg - true_neg
            # conf_mat[0, 0] += np.sum(true_pos)
            # conf_mat[0, 1] += np.sum(false_neg)
            # conf_mat[1, 0] += np.sum(false_pos)
            # conf_mat[1, 1] += np.sum(true_neg)
            # train_accuracy, train_sensitivity, train_specificity = \
            #     calc_metrics(conf_mat)
            # summary[epoch_i]['train_accuracy'] = train_accuracy
            # summary[epoch_i]['train_sensitivity'] = train_sensitivity
            # summary[epoch_i]['train_specificity'] = train_specificity
    
    
            # print('time: ' + str(time.clock() - epoch_time_ini)
            #       + ' - train_acc: ' + str(train_accuracy)
            #       + ' - train_sensitivity: ' + str(train_sensitivity)
            #       + ' - train_specificity: ' + str(train_specificity))
            #
            # fast_acc = ((epoch_true_pos + epoch_true_neg) /
            #             (epoch_positives + epoch_negatives))
            # fast_sen = epoch_true_pos / epoch_positives
            # fast_spe = epoch_true_neg / epoch_negatives
            #
            # print('FAST - time: ' + str(time.clock() - epoch_time_ini)
            #       + ' - fast_acc: ' + str(fast_acc)
            #       + ' - fast_sen: ' + str(fast_sen)
            #       + ' - fast_spe: ' + str(fast_spe))
            #
            # # epoch_log = ('train_acc ' + str(train_accuracy)
            # #              + ' train_sensitivity ' + str(
            # #     train_sensitivity)
            # #              + ' train_specificity ' + str(
            # #     train_specificity)
            # #              + '\n')
            # epoch_log = ('train_acc ' + str(fast_acc)
            #              + ' train_sensitivity ' + str(
            #     fast_sen)
            #              + ' train_specificity ' + str(
            #     fast_spe)
            #              + '\n')
            #
            # log_file.write(epoch_log)
    
            # TRAIN METRICS
            # TRAIN
            samples_it = 0
            conf_mat = np.zeros((2, 2))
    
            for X, y_true in train_eval_generator:
                model.train_on_batch(X, y_true, class_weight=None,
                                     sample_weight=None)
        
                y_pred = model.predict_on_batch(X)
                y_pred = np.sign(y_pred)
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
        
                samples_it += batch_size
                # End loop only condition
                if (samples_it + batch_size) > n_train:
                    break
    
            train_accuracy, train_sensitivity, train_specificity = \
                calc_metrics(conf_mat)
    
            cum_train_specificity += train_specificity
            cum_train_sensitivity += train_sensitivity
    
            # VAL METRICS
            samples_it = 0
    
            conf_mat_val = np.zeros((2, 2))
            for X, y_true in validation_generator:
                y_pred = model.predict_on_batch(X)
                y_pred = np.sign(y_pred)
                positives = np.maximum(y_true, 0.)
                pred_pos = np.maximum(y_pred, 0.)
                negatives = np.maximum(-y_true, 0.)
                pred_neg = np.maximum(-y_pred, 0.)
                true_pos = pred_pos * positives
                false_pos = pred_pos * negatives
                true_neg = pred_neg * negatives
                false_neg = pred_neg * positives
                conf_mat_val[0, 0] += np.sum(true_pos)
                conf_mat_val[0, 1] += np.sum(false_neg)
                conf_mat_val[1, 0] += np.sum(false_pos)
                conf_mat_val[1, 1] += np.sum(true_neg)
        
                samples_it += batch_size
                # End loop only condition
                if (samples_it + batch_size) > n_validation:
                    break
    
            val_accuracy, val_sensitivity, val_specificity = \
                calc_metrics(conf_mat_val)
    
            cum_val_specificity += val_specificity
            cum_val_sensitivity += val_sensitivity
    
            # AVERAGE METRICS
            avg_train_sen = cum_train_sensitivity / 10.
            avg_train_spe = cum_train_specificity / 10.
            avg_val_sen = cum_val_sensitivity / 10.
            avg_val_spe = cum_val_specificity / 10.
    
            # if (avg_train_sen > 0.6 and avg_train_spe >
            #     0.6 and avg_val_sen > 0.6 and
            #             avg_val_spe > 0.6):
            if True:
                with open('summary6.txt', 'a') as log_file_s:
                    log_file_s.write(
                        model_name
                        + ' avg_train_sen ' + str(avg_train_sen)
                        + ' avg_train_spe ' + str(avg_train_spe)
                        + ' avg_val_sen ' + str(avg_val_sen)
                        + ' avg_val_spe ' + str(avg_val_spe)
                        + '\n' + to_string(settings) + '\n')
    # RECORD ALL RESUTLS
    elif n_epoch > 0:
        for epoch_i in range(n_epoch):
            # epoch_time_ini = time.clock()
            summary[epoch_i] = OrderedDict([])
            # global epoch_y_pred
            # global epoch_y_true
            # epoch_y_pred = None
            # epoch_y_true = None
    
            # global epoch_true_pos
            # global epoch_true_neg
            # global epoch_positives
            # global epoch_negatives
            # epoch_true_pos = 0.
            # epoch_true_neg = 0.
            # epoch_positives = 0.
            # epoch_negatives = 0.
            # global epoch_conf_mat
            # epoch_conf_mat = np.zeros((2, 2))
            # print('Epoch ' + str(epoch_i) + '/' + str(n_epoch))
            # model.fit_generator(train_generator, n_train, 1,
            #                 verbose=1, callbacks=[
            #     # save_batch_results_callback,
            #     # save_epoch_results_callback,
            #     CSVLogger(log_file_name, separator=' ',
            #               append=True)
            #     ],
            #                 validation_data=validation_generator,
            #                 nb_val_samples=n_validation,
            #                 # class_weight=class_weight,
            #                 max_q_size=30,
            #                 nb_worker=1, pickle_safe=False)
            # epoch_train_time = time.clock() - epoch_time_ini
            # train_time += epoch_train_time
        
        # conf_mat = np.zeros((2, 2))
        # positives = np.maximum(epoch_y_true, 0.)
        # pred_pos = np.maximum(epoch_y_pred, 0.)
        # negatives = np.maximum(-epoch_y_true, 0.)
        # pred_neg = np.maximum(-epoch_y_pred, 0.)
        # true_pos = pred_pos * positives
        # false_pos = pred_pos - true_pos
        # true_neg = pred_neg * negatives
        # false_neg = pred_neg - true_neg
        # conf_mat[0, 0] += np.sum(true_pos)
        # conf_mat[0, 1] += np.sum(false_neg)
        # conf_mat[1, 0] += np.sum(false_pos)
        # conf_mat[1, 1] += np.sum(true_neg)
        # train_accuracy, train_sensitivity, train_specificity = \
        #     calc_metrics(conf_mat)
        # summary[epoch_i]['train_accuracy'] = train_accuracy
        # summary[epoch_i]['train_sensitivity'] = train_sensitivity
        # summary[epoch_i]['train_specificity'] = train_specificity
        
        
        # print('time: ' + str(time.clock() - epoch_time_ini)
        #       + ' - train_acc: ' + str(train_accuracy)
        #       + ' - train_sensitivity: ' + str(train_sensitivity)
        #       + ' - train_specificity: ' + str(train_specificity))
        #
        # fast_acc = ((epoch_true_pos + epoch_true_neg) /
        #             (epoch_positives + epoch_negatives))
        # fast_sen = epoch_true_pos / epoch_positives
        # fast_spe = epoch_true_neg / epoch_negatives
        #
        # print('FAST - time: ' + str(time.clock() - epoch_time_ini)
        #       + ' - fast_acc: ' + str(fast_acc)
        #       + ' - fast_sen: ' + str(fast_sen)
        #       + ' - fast_spe: ' + str(fast_spe))
        #
        # # epoch_log = ('train_acc ' + str(train_accuracy)
        # #              + ' train_sensitivity ' + str(
        # #     train_sensitivity)
        # #              + ' train_specificity ' + str(
        # #     train_specificity)
        # #              + '\n')
        # epoch_log = ('train_acc ' + str(fast_acc)
        #              + ' train_sensitivity ' + str(
        #     fast_sen)
        #              + ' train_specificity ' + str(
        #     fast_spe)
        #              + '\n')
        #
        # log_file.write(epoch_log)
    
    # TRAIN METRICS
            samples_it = 0
            # train_loss = 0.
            # batch_count = 0
            conf_mat = np.zeros((2, 2))
            for X, y_true in train_generator:

                model.train_on_batch(X, y_true, class_weight=None,
                                     sample_weight=None)
                y_pred = model.predict_on_batch(X)
                # train_loss += np.mean(np.maximum(1.-y_true*y_pred, 0.))
                y_pred = np.sign(y_pred)
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
    
                samples_it += batch_size
                # batch_count += 1
                # End loop only condition
                if (samples_it + batch_size) > n_train:
                    break
            train_accuracy, train_sensitivity, train_specificity = \
                calc_metrics(conf_mat)
            # train_loss = train_loss/batch_count
            # epoch_train_time = time.clock() - epoch_time_ini
            # train_time += epoch_train_time

            # VAL METRICS
            samples_it = 0
            # val_loss = 0.
            # batch_count = 0
            conf_mat_val = np.zeros((2, 2))
            for X, y_true in validation_generator:
                y_pred = model.predict_on_batch(X)
                # val_loss += np.mean(np.maximum(1. - y_true * y_pred, 0.))
                y_pred = np.sign(y_pred)
                positives = np.maximum(y_true, 0.)
                pred_pos = np.maximum(y_pred, 0.)
                negatives = np.maximum(-y_true, 0.)
                pred_neg = np.maximum(-y_pred, 0.)
                true_pos = pred_pos * positives
                false_pos = pred_pos * negatives
                true_neg = pred_neg * negatives
                false_neg = pred_neg * positives
                conf_mat_val[0, 0] += np.sum(true_pos)
                conf_mat_val[0, 1] += np.sum(false_neg)
                conf_mat_val[1, 0] += np.sum(false_pos)
                conf_mat_val[1, 1] += np.sum(true_neg)
    
                samples_it += batch_size
                # batch_count += 1
                # End loop only condition
                if (samples_it + batch_size) > n_validation:
                    break
    
            val_accuracy, val_sensitivity, val_specificity = \
                calc_metrics(conf_mat_val)
            
            # val_loss = val_loss / batch_count
        
            # PRINT METRICS
    
            print('Model: '+model_name+' - Epoch: ' + str(epoch_i)
                  # + ' - train_loss: ' + str(train_loss)
                  + ' - train_acc: ' + str(train_accuracy)
                  + ' - train_sensitivity: ' + str(train_sensitivity)
                  + ' - train_specificity: ' + str(train_specificity)
                  # + ' - val_loss: ' + str(val_loss)
                  + ' - val_acc: ' + str(val_accuracy)
                  + ' - val_sensitivity: ' + str(val_sensitivity)
                  + ' - val_specificity: ' + str(val_specificity))
    
            train_aux = [[model_name, epoch_i, #train_loss,
                          train_accuracy,
                          train_sensitivity,
                              train_specificity, 's', #val_loss,
                         val_accuracy, val_sensitivity, val_specificity]]
            
            with open(log_file_name, 'a') as log_file:
                log_file.write(to_string(train_aux)+'\n')

    #
    #
    # # print(conf_mat)
    # # print(sum(sum(conf_mat)))
    # # print(count_i)
    # train_accuracy = ((conf_mat[0, 0] + conf_mat[1, 1])
    #             / sum(sum(conf_mat)))
    # train_sensitivity = (conf_mat[0, 0]
    #                / (conf_mat[0, 0] + conf_mat[0, 1]))
    # train_specificity = (conf_mat[1, 1]
    #                / (conf_mat[1, 0] + conf_mat[1, 1]))
    # print('ACC: ' + str(train_accuracy))
    # print('SEN: ' + str(train_sensitivity))
    # print('SPE: ' + str(train_specificity))
    # # train_summary[epoch_it] = OrderedDict(
    # #     [('train_conf_mat', conf_mat), ('train_accuracy',
    # #                              train_accuracy),
    # #      ('train_sensitivity', train_sensitivity),
    # #      ('train_specificity', train_specificity)])
    # #
    #
    # # y_pred = model.predict_generator(val_eval_generator1, n_validation,
    # #                                  max_q_size=20,
    # #                                  nb_worker=1, pickle_safe=False)
    # #
    # # print(type(y_pred))
    # # print(len(y_pred))
    # # print(sum(y_pred))
    # # print(sum(y_pred>0))
    # # print(sum(y_pred < 0))
    # # print(sum(y_pred == 0))
    #
    # if val_eval_generator is not None:
    #     samples_it = 0
    #     conf_mat = np.zeros((2, 2))
    #     count_i = 0
    #     for X, y_true in val_eval_generator:
    #         # if temporal:
    #         #     model.reset_states()
    #
    #         y_pred = model.predict_on_batch(X)
    #         # print(len(y_pred))
    #         # print(len(y_true))
    #         # print((y_pred))
    #         y_pred = np.sign(y_pred)
    #         # print(len(y_pred))
    #         # print((y_pred))
    #
    #         for i in range(len(y_true)):
    #             count_i += 1
    #             if label_is_positive(y_true[i]):
    #                 if label_is_positive(y_pred[i]):
    #                     conf_mat[0, 0] += 1
    #                 else:
    #                     conf_mat[0, 1] += 1
    #             else:
    #                 if label_is_positive(y_pred[i]):
    #                     conf_mat[1, 0] += 1
    #                 else:
    #                     conf_mat[1, 1] += 1
    #
    #         samples_it += batch_size
    #         # End loop only condition
    #         if (samples_it + batch_size) > n_validation:
    #             break
    #
    #     # print(conf_mat)
    #     # print(sum(sum(conf_mat)))
    #     # print(count_i)
    #     val_accuracy = ((conf_mat[0, 0] + conf_mat[1, 1])
    #                       / sum(sum(conf_mat)))
    #     val_sensitivity = (conf_mat[0, 0]
    #                          / (conf_mat[0, 0] + conf_mat[0, 1]))
    #     val_specificity = (conf_mat[1, 1]
    #                          / (conf_mat[1, 0] + conf_mat[1, 1]))
    #     print('VAL_ACC: ' + str(val_accuracy))
    #     print('VAL_SEN: ' + str(val_sensitivity))
    #     print('VAL_SPE: ' + str(val_specificity))
    #     #
    #     # print(train_conf_mat)
    #     # print(type(train_conf_mat))
    #
    # exit(1)

    # else:
    #
    #     [status, trained_model, train_summary, settings] = single_train(
    #         model, train_generator, n_epoch, n_train, batch_size,
    #         temporal, class_weight,
    #         validation_generator=validation_generator,
    #         n_validation=n_validation, settings=settings)
    #     if check_status(status):
    #         msg = 'OK: Training process finished successfully'
    #         is_error = False
    #     else:
    #         msg = 'ERROR: Training process FAILED: ' + str(status)
    #         is_error = True
    #     report_event(msg, is_error=is_error)
    #
    #     if settings is not None:
    #         settings = define_settings(settings, final_status=status)
    # train_summary = None
    if settings is not None:
        settings = define_settings(
            settings,
            # train_accuracy=train_accuracy,
            # train_sensitivity=train_sensitivity,
            # train_specificity=train_specificity,
            training_time=train_time,
            # validation_accuracy=val_accuracy,
            # validation_sensitivity=val_sensitivity,
            # validation_specificity=val_specificity
        )
        
    with open(log_file_name, 'a') as log_file:
        log_file.write(to_string(settings))
    # print(settings)
    # exit(1)
    return [model, summary, settings, epoch_count]


def single_train(model, train_generator, n_epoch, n_train,
                 batch_size, temporal, class_weight,
                 validation_generator=None,
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
            reset_state = additional_info['reset_state']
            # if temporal and reset_state:
            #     model.reset_states()
            #     print('Reset')
            #     print(current_patinet)
            #     print(current_file)
            # else:
            #     print('continue')
            #     print(current_patinet)
            #     print(current_file)
            try:
                model.train_on_batch(X, y, class_weight=class_weight)
            except Exception as e:
                status = str(repr(e))
                msg = ('ERROR:  Train failed for epoch '
                       + str(epoch_it) + ' during patient: '
                       + current_patinet + ' and status: ' + status)
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
                model, train_generator, batch_size, n_train, temporal,
                validation_generator=validation_generator,
                n_validation=n_validation)
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
        # train_metrics = metrics_calc(train_conf_mat)
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
                # val_metrics = metrics_calc(val_conf_mat)
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
        model, train_generator, batch_size, n_train, temporal,
        validation_generator=validation_generator,
        n_validation=n_validation)


# EOF


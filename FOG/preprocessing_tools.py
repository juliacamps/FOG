"""IO functions for FOG deep-learning project"""

# Authors: Julia Camps <julia.camps.sereix@est.fib.upc.edu>
# Created on: 06/10/2016 17:22

import numpy as np
import random as rd
from collections import Counter

from FOG.definitions import get_patient_partition
from FOG.definitions import get_data_structure
from FOG.definitions import define_settings
from FOG.definitions import get_inter_delimiter
from FOG.definitions import label_is_valid
from FOG.definitions import get_positive_label
from FOG.definitions import get_negative_label
from FOG.definitions import get_undefined_label
# from FOG.definitions import get_epsilon
# from FOG.definitions import label_is_positive

# from FOG.utils import substract_mean
from FOG.utils import split_data
from FOG.utils import parse_value
from FOG.utils import fft_window

from FOG.experiment_conf import get_rotate_range
# from FOG.experiment_conf import get_shift_augment
# from FOG.experiment_conf import get_filter_threshold
# from FOG.experiment_conf import get_rotate_augment

from FOG.io_functions import get_data_property
from FOG.io_functions import report_event
from FOG.io_functions import read_data_file

# import matplotlib.pyplot as plt


def generate_batches(data_structure, window_size, batch_size,
                     temporal, global_std, shift_augmentation,
                     rotate_augmentation, pure_threshold, percent_throw_no_fog,
                     pos_threshold, cutting):
    """Create data set generator"""
    rotate_range = get_rotate_range()
    shift_count = shift_augmentation + 1
    rotate_count = rotate_augmentation + 1

    while 1:
        # pos_batch_count = 0
        # instance_count = 0
        # batch_count = 0
        # y_cum = 0
        # # reset_count = 0
        # # continue_count = 0
        # batch_X = []
        # batch_Y = []
        # # batch_Y_orig = []
        # batch_it = 0
        epoch_data = [file for patient in list(
            data_structure.items()) for file in patient[1]]
        rd.shuffle(epoch_data)
        # X_total = None
        # X_total_or = None
        # X_total_raw = None
        # y_true_total = np.array([])

        for shift_it in range(shift_count):
            for rotate_it in range(rotate_count):
                for file_data in epoch_data:
                    shift = int(
                        round(window_size
                              * np.random.uniform(low=0.0,
                                                  high=1.0)))
                    if rotate_it+1 < rotate_count:
                        rotate_mat = random_rot(rotate_range)
                    else:
                        rotate_mat = np.identity(3)
                    # print('shift ' + str(shift_it))
                    # print(shift)
                    # print('rotate ' + str(rotate_it))
                    # print(rotate_mat)
                    
                    # if temporal:
                    #     batch_X = []
                    #     batch_Y = []
                    #     # batch_Y_orig = []
                    #     batch_it = 0

                    batch_X = []
                    batch_Y = []
                    batch_it = 0
                    ant_X = None
                    for X_raw, y_raw, y_orig in get_windows(
                            file_data, window_size, shift):
                        # instance_count += X_raw.shape[0]
                        y, valid = check_label(y_raw,
                                               pure_threshold)
                        act_X = preprocess_data(X_raw, global_std,
                                                rotate_mat,
                                                cutting_mode=cutting)
                        # act_X = X_new.copy()
                        if ant_X is None:
                            ant_X = act_X.copy()
                            # ant_X2 = X.copy()
                        if valid and window_size == act_X.shape[0]:
                            # y_true_total = np.concatenate(
                            #         (y_true_total,
                            #          np.full(window_size, y,
                            #                  dtype=int)
                            #          ), axis=0)
                            # if X_total_raw is None:
                            #     X_total_raw = X
                            # else:
                            #     X_total_raw = np.concatenate((X_total_raw, X))
                            # if X_total_or is None:
                            #     X_total_or = preprocess_data(X, global_std,
                            #                     np.identity(3))
                            # else:
                            #     X_total_or = np.concatenate((X_total_or, preprocess_data(X, global_std,
                            #                     np.identity(3))))
    
                            X_new = add_time_dep(act_X, ant_X,
                                                 cutting)
                            # X_new = np.concatenate(
                            #     (X_new, act_X - ant_X),
                            #     axis=1)
    
                            # ant_X2 = ant_X1.copy()
                            # if X_total is None:
                            #     X_total = X
                            # else:
                            #     X_total = np.concatenate((X_total, X))
    
                            # batch_Y_orig.append(y_orig)
    
                            # if not temporal and
                            if percent_throw_no_fog > 0:
                                if percent_throw_no_fog < rd.random():
                                    batch_X.append(X_new)
                                    batch_Y.append(y)
                                    batch_it += 1
                                    
                            else:
                                batch_X.append(X_new)
                                batch_Y.append(y)
                                batch_it += 1
                                
                            if batch_it == batch_size:
                                # y_cum_part = sum(np.asarray(
                                #     batch_Y) > 0)
                                # if pos_threshold:
                                #     if y_cum_part > 0:
                                #         yield (np.asarray(batch_X),
                                #                np.asarray(batch_Y))
                                #         # pos_batch_count += 1
                                #         # y_cum += y_cum_part
                                #         # batch_count += 1
                                # else:
                                
                                yield (np.asarray(batch_X),
                                       np.asarray(batch_Y))
                                # if y_cum_part > 0:
                                #     pos_batch_count += 1
                                # y_cum += y_cum_part
                                # batch_count += 1
                                batch_X = []
                                batch_Y = []
                                # batch_Y_orig = []
                                batch_it = 0
    
                        ant_X = act_X.copy()
                    
                 #    file_data_generator = get_file_generator(file_data, window_size, shift, rotate_mat,
                 # global_std, cutting, pure_threshold, percent_throw_no_fog, batch_size)
                 #    for patient_data in file_data_generator:
                 #        yield (patient_data)

                    
                        # elif temporal:
                        #     batch_it = 0
                        #     batch_X = []
                        #     batch_Y = []
                            
                            # batch_Y_orig = []
                #     #ACC
                #     # Plot transformation
                #     ax = X_total[:, 3]
                #     ay = X_total[:, 4]
                #     az = X_total[:, 5]
                #
                #     time_data = np.linspace(0, (
                #     X_total.shape[0] * (1 / 100)),
                #                             X_total.shape[0])
                #
                #     # f1 = plt.figure()
                #     # ax1 = f1.add_subplot(111)
                #     # ax2 = f1.add_subplot(112)
                #     # ax3 = f1.add_subplot(113)
                #     # ax4 = f1.add_subplot(114)
                #     f1, (ax1, ax2, ax3) = plt.subplots(3,
                #                                        sharex=True, sharey=True)
                #
                #     ax1.set_title('ACC-Transformed')
                #     ax1.set_xlabel('time')
                #
                #     # ax1.plot(time_data, y_orig_total, color='y',
                #     #          label='y_orig')
                #     ax1.plot(time_data, ax, color='r', label='accX')
                #     ax2.plot(time_data, ay, color='g', label='accY')
                #     ax3.plot(time_data, az, color='b', label='accZ')
                #     # ax4.plot(time_data, y_true_total, color='c',
                #     #          label='y_true')
                #
                #     # leg = f1.legend()
                #
                #     # Plot original
                #     ax = X_total_or[:, 3]
                #     ay = X_total_or[:, 4]
                #     az = X_total_or[:, 5]
                #
                #     time_data = np.linspace(0, (
                #         X_total_or.shape[0] * (1 / 100)),
                #                             X_total_or.shape[0])
                #
                #     # f2 = plt.figure()
                #     # ax1 = f2.add_subplot(111)
                #     # ax2 = f2.add_subplot(121)
                #     # ax3 = f2.add_subplot(131)
                #     # ax4 = f2.add_subplot(141)
                #     f2, (ax1, ax2, ax3) = plt.subplots(3,
                #     sharex=True, sharey=True)
                #
                #     ax1.set_title('ACC-Original')
                #     ax1.set_xlabel('time')
                #
                #     # ax1.plot(time_data, y_orig_total, color='y',
                #     #          label='y_orig')
                #     ax1.plot(time_data, ax, color='r', label='accX')
                #     ax2.plot(time_data, ay, color='g', label='accY')
                #     ax3.plot(time_data, az, color='b', label='accZ')
                #     # ax4.plot(time_data, y_true_total, color='c',
                #     #          label='y_true')
                #
                #     # leg = f2.legend()
                #
                #     # Plot Raw
                #     ax = X_total_raw[:, 3]
                #     ay = X_total_raw[:, 4]
                #     az = X_total_raw[:, 5]
                #
                #     time_data = np.linspace(0, (
                #         X_total_raw.shape[0] * (1 / 100)),
                #                             X_total_raw.shape[0])
                #
                #     # f2 = plt.figure()
                #     # ax1 = f2.add_subplot(111)
                #     # ax2 = f2.add_subplot(121)
                #     # ax3 = f2.add_subplot(131)
                #     # ax4 = f2.add_subplot(141)
                #     f3, (ax1, ax2, ax3) = plt.subplots(3,
                #                                        sharex=True,
                #                                        sharey=True)
                #
                #     ax1.set_title('ACC-Raw')
                #     ax1.set_xlabel('time')
                #
                #     # ax1.plot(time_data, y_orig_total, color='y',
                #     #          label='y_orig')
                #     ax1.plot(time_data, ax, color='r', label='accX')
                #     ax2.plot(time_data, ay, color='g', label='accY')
                #     ax3.plot(time_data, az, color='b', label='accZ')
                #     # ax4.plot(time_data, y_true_total, color='c',
                #     #          label='y_true')
                #
                #     # leg = f2.legend()
                #
                #     #GIRO
                #     # Plot transformation
                #     ax = X_total[:, 0]
                #     ay = X_total[:, 1]
                #     az = X_total[:, 2]
                #
                #     time_data = np.linspace(0, (
                #         X_total.shape[0] * (1 / 100)),
                #                             X_total.shape[0])
                #
                #     # f1 = plt.figure()
                #     # ax1 = f1.add_subplot(111)
                #     # ax2 = f1.add_subplot(112)
                #     # ax3 = f1.add_subplot(113)
                #     # ax4 = f1.add_subplot(114)
                #     f4, (ax1, ax2, ax3) = plt.subplots(3,
                #                                        sharex=True,
                #                                        sharey=True)
                #
                #     ax1.set_title('GIRO-Transformed')
                #     ax1.set_xlabel('time')
                #
                #     # ax1.plot(time_data, y_orig_total, color='y',
                #     #          label='y_orig')
                #     ax1.plot(time_data, ax, color='r', label='accX')
                #     ax2.plot(time_data, ay, color='g', label='accY')
                #     ax3.plot(time_data, az, color='b', label='accZ')
                #     # ax4.plot(time_data, y_true_total, color='c',
                #     #          label='y_true')
                #
                #     # leg = f1.legend()
                #
                #     # Plot original
                #     ax = X_total_or[:, 0]
                #     ay = X_total_or[:, 1]
                #     az = X_total_or[:, 2]
                #
                #     time_data = np.linspace(0, (
                #         X_total_or.shape[0] * (1 / 100)),
                #                             X_total_or.shape[0])
                #
                #     # f2 = plt.figure()
                #     # ax1 = f2.add_subplot(111)
                #     # ax2 = f2.add_subplot(121)
                #     # ax3 = f2.add_subplot(131)
                #     # ax4 = f2.add_subplot(141)
                #     f5, (ax1, ax2, ax3) = plt.subplots(3,
                #                                        sharex=True,
                #                                        sharey=True)
                #
                #     ax1.set_title('GIRO-Original')
                #     ax1.set_xlabel('time')
                #
                #     # ax1.plot(time_data, y_orig_total, color='y',
                #     #          label='y_orig')
                #     ax1.plot(time_data, ax, color='r', label='accX')
                #     ax2.plot(time_data, ay, color='g', label='accY')
                #     ax3.plot(time_data, az, color='b', label='accZ')
                #     # ax4.plot(time_data, y_true_total, color='c',
                #     #          label='y_true')
                #
                #     # leg = f2.legend()
                #
                #     # Plot Raw
                #     ax = X_total_raw[:, 0]
                #     ay = X_total_raw[:, 1]
                #     az = X_total_raw[:, 2]
                #
                #     time_data = np.linspace(0, (
                #         X_total_raw.shape[0] * (1 / 100)),
                #                             X_total_raw.shape[0])
                #
                #     # f2 = plt.figure()
                #     # ax1 = f2.add_subplot(111)
                #     # ax2 = f2.add_subplot(121)
                #     # ax3 = f2.add_subplot(131)
                #     # ax4 = f2.add_subplot(141)
                #     f3, (ax1, ax2, ax3) = plt.subplots(3,
                #                                        sharex=True,
                #                                        sharey=True)
                #
                #     ax1.set_title('GIRO-Raw')
                #     ax1.set_xlabel('time')
                #
                #     # ax1.plot(time_data, y_orig_total, color='y',
                #     #          label='y_orig')
                #     ax1.plot(time_data, ax, color='r', label='accX')
                #     ax2.plot(time_data, ay, color='g', label='accY')
                #     ax3.plot(time_data, az, color='b', label='accZ')
                #     # ax4.plot(time_data, y_true_total, color='c',
                #     #          label='y_true')
                #
                #     # leg = f2.legend()
                #
                #     #MAGNETO
                #     # Plot transformation
                #     ax = X_total[:, 6]
                #     ay = X_total[:, 7]
                #     az = X_total[:, 8]
                #
                #     time_data = np.linspace(0, (
                #         X_total.shape[0] * (1 / 100)),
                #                             X_total.shape[0])
                #
                #     # f1 = plt.figure()
                #     # ax1 = f1.add_subplot(111)
                #     # ax2 = f1.add_subplot(112)
                #     # ax3 = f1.add_subplot(113)
                #     # ax4 = f1.add_subplot(114)
                #     f4, (ax1, ax2, ax3) = plt.subplots(3,
                #                                        sharex=True,
                #                                        sharey=True)
                #
                #     ax1.set_title('MAGNETO-Transformed')
                #     ax1.set_xlabel('time')
                #
                #     # ax1.plot(time_data, y_orig_total, color='y',
                #     #          label='y_orig')
                #     ax1.plot(time_data, ax, color='r', label='accX')
                #     ax2.plot(time_data, ay, color='g', label='accY')
                #     ax3.plot(time_data, az, color='b', label='accZ')
                #     # ax4.plot(time_data, y_true_total, color='c',
                #     #          label='y_true')
                #
                #     # leg = f1.legend()
                #
                #     # Plot original
                #     ax = X_total_or[:, 6]
                #     ay = X_total_or[:, 7]
                #     az = X_total_or[:, 8]
                #
                #     time_data = np.linspace(0, (
                #         X_total_or.shape[0] * (1 / 100)),
                #                             X_total_or.shape[0])
                #
                #     # f2 = plt.figure()
                #     # ax1 = f2.add_subplot(111)
                #     # ax2 = f2.add_subplot(121)
                #     # ax3 = f2.add_subplot(131)
                #     # ax4 = f2.add_subplot(141)
                #     f5, (ax1, ax2, ax3) = plt.subplots(3,
                #                                        sharex=True,
                #                                        sharey=True)
                #
                #     ax1.set_title('MAGNETO-Original')
                #     ax1.set_xlabel('time')
                #
                #     # ax1.plot(time_data, y_orig_total, color='y',
                #     #          label='y_orig')
                #     ax1.plot(time_data, ax, color='r', label='accX')
                #     ax2.plot(time_data, ay, color='g', label='accY')
                #     ax3.plot(time_data, az, color='b', label='accZ')
                #     # ax4.plot(time_data, y_true_total, color='c',
                #     #          label='y_true')
                #
                #     # leg = f2.legend()
                #
                #     # Plot Raw
                #     ax = X_total_raw[:, 6]
                #     ay = X_total_raw[:, 7]
                #     az = X_total_raw[:, 8]
                #
                #     time_data = np.linspace(0, (
                #         X_total_raw.shape[0] * (1 / 100)),
                #                             X_total_raw.shape[0])
                #
                #     # f2 = plt.figure()
                #     # ax1 = f2.add_subplot(111)
                #     # ax2 = f2.add_subplot(121)
                #     # ax3 = f2.add_subplot(131)
                #     # ax4 = f2.add_subplot(141)
                #     f3, (ax1, ax2, ax3) = plt.subplots(3,
                #                                        sharex=True,
                #                                        sharey=True)
                #
                #     ax1.set_title('MAGNETO-Raw')
                #     ax1.set_xlabel('time')
                #
                #     # ax1.plot(time_data, y_orig_total, color='y',
                #     #          label='y_orig')
                #     ax1.plot(time_data, ax, color='r', label='accX')
                #     ax2.plot(time_data, ay, color='g', label='accY')
                #     ax3.plot(time_data, az, color='b', label='accZ')
                #     # ax4.plot(time_data, y_true_total, color='c',
                #     #          label='y_true')
                #
                #     # leg = f2.legend()
                #
                #     plt.show()
                #     exit(1)
                #     break
                # return 1
                                    
        # n_samples = batch_count * batch_size
        # print('Total number of samples in all data: ')
        # print(instance_count/batch_size)
        # print('Num Batches: ' + str(batch_count))
        # print('Num Positive Batches: ' + str(pos_batch_count))
        # print('Percent of pos_batch: ' + str(
        #     pos_batch_count/batch_count))
        # print('Available number of samples: ' + str(n_samples))
        # print('Percentage of FOG:')
        # print(y_cum / n_samples)
        # print('Percentage of lost samples:')
        # print(1 - (n_samples * window_size) / instance_count)
        # print((instance_count - n_samples * window_size)/instance_count)
        # print('Discarded only for threshold filtering: '
        #       + str(discarded))
        # print('Reset: ' + str(reset_count))
        # print('Continue: ' + str(continue_count))
        
        

    #     if percent_throw_no_fog >= 0:
    #         print('    '+"'"+'n_train'+"'"+': '+str(n_samples)
    #               +',\n    '+"'"+'percent_pos_train'+"'"+': '+str(y_cum / n_samples)
    #               +',')
    #     else:
    #         print('    '+"'"+'n_validation'+"'"+': '+str(n_samples)
    #               +',\n    '+"'"+'percent_pos_val'+"'"+': '+str(y_cum / n_samples)
    #               +',')
    #     break
    # return 1

def add_time_dep(act_X, ant_X, cutting):
    """"""
    if cutting > 1:
        act_X_parts = np.split(act_X, cutting)
        ant_X_parts = np.split(ant_X, cutting)
        X_new = np.concatenate((act_X_parts[0], act_X_parts[0] -
                                ant_X_parts[-1]), axis=1)
        for i in range(1, cutting, 1):
            new_part = np.concatenate((act_X_parts[i], act_X_parts[i]
                                       - act_X_parts[i-1]), axis=1)
            X_new = np.concatenate((X_new, new_part), axis=0)
    else:
        X_new = np.concatenate((act_X, act_X - ant_X), axis=1)
    return X_new

def get_file_generator(file_data, window_size, shift, rotate_mat,
                 global_std, cutting, pure_threshold,
                 percent_throw_no_fog, batch_size,
                       get_raw_data=False):
    """"""
    batch_X = []
    batch_Y = []
    batch_X_raw = []
    batch_Y_raw = []
    batch_it = 0
    ant_X1 = None
    for X_raw, y_raw, y_orig in get_windows(
            file_data, window_size, shift):
        y, valid = check_label(y_raw, pure_threshold)
        X_new = preprocess_data(X_raw, global_std,
                            rotate_mat,
                            cutting_mode=cutting)
        act_X = X_new.copy()
        if ant_X1 is None:
            ant_X1 = act_X.copy()
        if valid and window_size == X_new.shape[0]:
            X_new = np.concatenate((X_new, act_X - ant_X1),
                               axis=1)
            if percent_throw_no_fog > 0:
                if percent_throw_no_fog < rd.random():
                    batch_X.append(X_new)
                    batch_Y.append(y)
                    batch_it += 1
                    if get_raw_data:
                        batch_X_raw.append(X_raw)
                        batch_Y_raw.append(y_raw)
            else:
                batch_X.append(X_new)
                batch_Y.append(y)
                batch_it += 1
                if get_raw_data:
                    batch_X_raw.append(X_raw)
                    batch_Y_raw.append(y_raw)
            if batch_it == batch_size:
                if get_raw_data:
                    yield (np.asarray(batch_X), np.asarray(
                        batch_Y), np.asarray(batch_X_raw),
                           np.asarray(batch_Y_raw))
                else:
                    yield (np.asarray(batch_X), np.asarray(batch_Y))
                batch_X = []
                batch_Y = []
                batch_X_raw = []
                batch_Y_raw = []
                batch_it = 0
    
        ant_X1 = act_X.copy()


def get_windows(file_data, window_size, shift):
    """"""
    data = []
    for it in range(shift, file_data.shape[0], window_size):
        if it + window_size <= file_data.shape[0]:
            data.append(split_data(file_data[it:(it + window_size)]))
    return data


def parse_line(line):
    """"""
    return [parse_value(line_part, force=True) for line_part in
            line.split(
        get_inter_delimiter())]


# def parse_y_orig(y_orig_raw):
#     """"""
#     return Counter(y_orig_raw)


def get_patient_split(problem):
    """Split patients between train, validation and test groups
        
    Return
    ------
    train_patient, val_patient, test_patient : str array-like
    
    """
    [train, validation, test] = get_patient_partition(problem=problem)
    patient_data = get_data_structure(problem=problem)
    train_data = {patient_name: patient_path
                  for patient_name, patient_path in
                  patient_data.items() if patient_name in train}
    val_data = {patient_name: patient_path
                for patient_name, patient_path in
                patient_data.items() if patient_name in validation}
    test_data = {patient_name: patient_path
                 for patient_name, patient_path in
                 patient_data.items() if patient_name in test}
    return [train_data, val_data, test_data]


def preprocess_data(X, global_std, rotation, cutting_mode):
    """Center and normalize data.
    
    Parameters
    ----------
    X : array-like, shape=[n_sample, n_features]
    rotation : square-matrix
    """
    X = X / global_std

    X = np.concatenate((
        np.dot(rotation, np.asarray(X[:, :3]).T).T,
        np.dot(rotation, np.asarray(X[:, 3:6]).T).T,
        np.dot(rotation, np.asarray(X[:, 6:]).T).T), axis=1)
    if cutting_mode > 1:
        X_parts = np.split(X, cutting_mode)
        X_new = fft_window(X_parts[0])
        for i in range(1, cutting_mode, 1):
            X_new = np.concatenate((X_new, fft_window(X_parts[i])))
    else:
        X_new = fft_window(X)
    return X_new


def check_label(Y, filter_th=0.0):
    """Check correctness of labels"""
    y_count = Counter(Y)
    label = None
    if get_positive_label() in y_count:
        if Y.shape[0] / y_count[get_positive_label()] > filter_th:
            label = get_positive_label()
    elif get_undefined_label() in y_count:
        label = get_undefined_label()
    elif get_negative_label() in y_count:
        if Y.shape[0] / y_count[get_negative_label()] > filter_th:
            label = get_negative_label()
    if label is None:
        label = get_undefined_label()
    return label, label_is_valid(label)
    

def random_rot(range):
    """Return a random rotation matrix"""
    theta = rd.uniform(-range[0], range[0])
    Rx = np.asarray([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)]])
    theta = rd.uniform(-range[1], range[1])
    Ry = np.matrix([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)]])
    theta = rd.uniform(-range[2], range[2])
    Rz = np.matrix([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    R = np.dot(np.dot(Rx, Ry), Rz)
    return R


def get_dataset(patients):
    """"""
    dataset = {}
    for patient_name, patient_files in list(patients.items()):
        dataset[patient_name] = []
        for patient_file in patient_files:
            dataset[patient_name].append(read_data_file(patient_file))
    return dataset


def get_data_files(patients):
    """"""
    for patient_name, patient_files in list(patients.items()):
        for patient_file in patient_files:
            yield (patient_name, patient_file, read_data_file(patient_file))


def get_prediction_generator(file_data, window_size, batch_size,
                             cutting, pure_threshold,
                             percent_throw_no_fog, problem):
    """"""
    rotate_mat = np.identity(3)
    shift = 0
    global_std = get_data_property(problem=problem,
                                   property_key='std')
    file_data_generator = get_file_generator(file_data, window_size,
                                             shift, rotate_mat,
                                             global_std, cutting,
                                             pure_threshold,
                                             percent_throw_no_fog,
                                             batch_size,
                                             get_raw_data=True)
    for patient_data in file_data_generator:
        yield (patient_data)

def get_generator(train_data, window_size, batch_size, temporal,
                  problem, shift_augmentation, rotate_augmentation,
                    cutting, pure_threshold,
                  validation_data=None,
                  settings=None, percent_throw_no_fog=0.,
                  pos_threshold=False):
    """"""

    global_std = get_data_property(problem=problem,
                                   property_key='std')
    # if pure_threshold is None:
    #     pure_threshold = get_filter_threshold()
    
    if settings is not None:
        settings = define_settings(
            settings, global_std=global_std,
            augment_shift=shift_augmentation,
            augment_rotate=rotate_augmentation,
            pure_threshold=pure_threshold)
    
    # train_data = get_dataset(train_patient)
    train_generator = generate_batches(
        train_data, window_size, batch_size, temporal, global_std,
        shift_augmentation, rotate_augmentation, pure_threshold,
        percent_throw_no_fog, pos_threshold, cutting)
    # report_event('OK: Train Generator setup')
        
    if validation_data is None:
        validation_generator = None
        # report_event('WARNING: No validation data configured')
    else:
        # validation_data = get_dataset(validation_patient)
        validation_generator = generate_batches(
            validation_data, window_size, batch_size, temporal,
            global_std, 0, 0, pure_threshold, 0, False, cutting)
        # report_event('OK: Validation Generator setup')
        
    return train_generator, validation_generator, settings
    

if __name__ == '__main__':
    problem = 'fog'
    reproducibility = False
    seed = 77
    if reproducibility:
        np.random.seed(seed)
        rd.seed(seed) # False,
    [train_patient, val_patient, test_patient] = get_patient_split(
        problem=problem)
    train_data = get_dataset(train_patient)
    validation_data = get_dataset(val_patient)
    test_data = get_dataset(test_patient)

    data_freq = 100
    n_shift = 3
    n_rotate = 1
    temporal = False
    pos_threshold = False
    configurations = [
        {
            'window_time': 2.56,
            'batch_size': 16,
            'pure_threshold': 0.5,
            'percent_throw_no_fog': 0.
        },
        
        # {
        #     'window_time': 2.56,
        #     'batch_size': 16,
        #     'pure_threshold': 0.5,
        #     'percent_throw_no_fog': 0.5
        # },
        # {
        #     'window_time': 2.56,
        #     'batch_size': 32,
        #     'pure_threshold': 0.5,
        #     'percent_throw_no_fog': 0.
        # },
        # {
        #     'window_time': 5.12,
        #     'batch_size': 16,
        #     'pure_threshold': 0.1,
        #     'percent_throw_no_fog': 0.
        # },
        # {
        #     'window_time': 5.12,
        #     'batch_size': 16,
        #     'pure_threshold': 0.25,
        #     'percent_throw_no_fog': 0.
        # },
        # {
        #     'window_time': 5.12,
        #     'batch_size': 16,
        #     'pure_threshold': 0.5,
        #     'percent_throw_no_fog': 0.
        # },
        # {
        #     'window_time': 5.12,
        #     'batch_size': 16,
        #     'pure_threshold': 0.1,
        #     'percent_throw_no_fog': 0.5
        # },
        # {
        #     'window_time': 5.12,
        #     'batch_size': 16,
        #     'pure_threshold': 0.25,
        #     'percent_throw_no_fog': 0.5
        # },
        # {
        #     'window_time': 5.12,
        #     'batch_size': 16,
        #     'pure_threshold': 0.5,
        #     'percent_throw_no_fog': 0.5
        # },
        # {
        #     'window_time': 10.24,
        #     'batch_size': 16,
        #     'pure_threshold': 0.1,
        #     'percent_throw_no_fog': 0.
        # }
    ]
    for conf in configurations:
        window_time = conf['window_time']
        batch_size = conf['batch_size']
        pure_threshold = conf['pure_threshold']
        percent_throw_no_fog = conf['percent_throw_no_fog']
        window_size = int(window_time * data_freq)

        print('('+"'"+'f'+str(data_freq)+'_w'+(str(
            window_time).replace('.','_'))+'_b'+str(
            batch_size)+'_s'+str(n_shift)+'_r'+str(
            n_rotate)+'_p'+str(
            pure_threshold)+"'"
              +', {\n    '+"'"+'augment_shift'+"'"+': '+str(
            n_shift)+',\n    '+"'"+'augment_rotate'+"'"+': '
              +str(n_rotate)+',\n    '+"'"+'data_freq'+"'"+': '
              +str(data_freq)+',\n    '+"'"+'is_temporal'+"'"+': '
              +str(temporal)+',\n    '+"'"+'window_time'+"'"+': '
              +str(window_time)+',\n    '+"'"+'batch_size'+"'"+': '
              +str(batch_size)+',\n    '+"'"+'pure_threshold'+"'"+': '
              + str(pure_threshold) + ',\n    ' + "'" + 'pos_threshold' + "'" + ': '
              +str(pos_threshold)+',\n    '+"'"+'percent_throw_no_fog'+"'"+': '
              +str(percent_throw_no_fog)+',')
        get_generator(
            train_data, window_size, batch_size, temporal,
            problem, validation_data=None,
            shift_augmentation=n_shift,
            rotate_augmentation=n_rotate,
            pure_threshold=pure_threshold, settings=None,
            percent_throw_no_fog=percent_throw_no_fog,
            pos_threshold=pos_threshold, cutting=1)
        # print('VAL')
        get_generator(
            validation_data, window_size, batch_size,
            temporal,
            problem, validation_data=None,
            shift_augmentation=0, rotate_augmentation=0,
            pure_threshold=pure_threshold, settings=None,
            percent_throw_no_fog=-1, pos_threshold=False,
            cutting=1)
        print(
            '    ' + "'" + 'n_test' + "'" + ': ' +
            str(0) + ',\n    ' + "'" +
            'percent_pos_test' + "'" + ': ' + str(0)
            + '\n}),')
                
    print('END')

# EOF

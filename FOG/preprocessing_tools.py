"""IO functions for FOG deep-learning project"""

# Authors: Julia Camps <julia.camps.sereix@est.fib.upc.edu>
# Created on: 06/10/2016 17:22

import numpy as np
import random as rd
import pandas as pd
from collections import Counter

from FOG.utils import substract_mean
from FOG.definitions import get_patient_partition
from FOG.definitions import get_data_structure
from FOG.experiment_conf import get_rotate_range
from FOG.definitions import label_is_valid
from FOG.io_functions import get_data_property
from FOG.experiment_conf import get_shift_augment
from FOG.experiment_conf import get_filter_threshold
from FOG.experiment_conf import get_rotate_augment
from FOG.io_functions import report_event
from FOG.definitions import define_settings
from FOG.definitions import get_inter_delimiter
from FOG.utils import split_data


def generate_batches(data_structure, window_size, batch_size,
                     temporal, global_std, shift_augmentation,
                     rotate_augmentation, threshold):
    """Create data set generator"""
    rotate_range = get_rotate_range()
    shift_count = 1 + shift_augmentation
    rotate_count = 1 + rotate_augmentation

    while 1:
        batch_count = 0
        y_cum = 0
        batch_X = []
        batch_Y = []
        batch_Y_orig = []
        batch_it = 0
        for patient_name, patient_data in data_structure.items():
            # Augmentation setup: different for each patient and epoch
            shift_indexes = [0]
            rotate = [np.identity(3)]
            if shift_count > 1:
                for rand_shift in np.random.uniform(
                        low=0.0, high=1.0,
                        size=(shift_count - 1)):
                    shift_indexes.append(
                        int(round(window_size * rand_shift)))
    
            if rotate_count > 1:
                for i in range(rotate_count - 1):
                    rotate.append(random_rot(rotate_range))
            
            # Iteration over files
            for file_path in patient_data:
                # For every augmentation
                for it_shift in range(shift_count):
                    shift = shift_indexes[it_shift]
                    for it_rot in range(rotate_count):
                        rotate_mat = rotate[it_rot]
                        if temporal:
                            batch_X = []
                            batch_Y = []
                            batch_Y_orig = []
                            batch_it = 0
                        for data in pd.read_csv(
                                file_path, delimiter=get_inter_delimiter(),
                         dtype=float, chunksize=window_size,
                                na_filter=False, skiprows=shift,
                         header=None):
                            X, y, y_orig = split_data(data.as_matrix())
                            y, valid = check_label(y, threshold)
                            if valid and window_size == X.shape[0]:
                                X = preprocess_data(X, global_std,
                                                    rotate_mat)
                                batch_X.append(X)
                                batch_Y.append(y)
                                batch_Y_orig.append(y_orig)
                                batch_it += 1
                                if batch_it == batch_size:
                                    yield ((np.asarray(batch_X),
                                           np.asarray(batch_Y),
                                           np.asarray(batch_Y_orig)),
                                           {'patient': patient_name,
                                            'file': file_path})
                                    y_cum += sum(np.asarray(batch_Y))
                                    batch_count += 1
                                    batch_X = []
                                    batch_Y = []
                                    batch_Y_orig = []
                                    batch_it = 0
                            elif temporal:
                                batch_it = 0
                                batch_X = []
                                batch_Y = []
                                batch_Y_orig = []
                                # model_.reset_states()
            
            # model_.reset_states()
    #     n_samples = batch_count * batch_size
    #     # print('Total number of samples in all data: ')
    #     # print(aux_count)
    #     print('Num Batches: ' + str(batch_count))
    #     print('Available number of samples: ' + str(
    #         n_samples))
    #     print('Percentage of FOG:')
    #     print(y_cum / n_samples)
    #     # print('Percentage of lost samples:')
    #     # print(1 - (n_samples * window_size) / aux_count)
    #     # print('Discarded only for threshold filtering: '
    #     #       + str(discarded))
    #     break
    # return 1


def parse_y_orig(y_orig_raw):
    """"""
    return Counter(y_orig_raw)


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


def preprocess_data(X, global_std, rotation):
    """Center and normalize data.
    
    Parameters
    ----------
    X : array-like, shape=[n_sample, n_features]
    rotation : square-matrix
    """
    X /= global_std
    X = np.concatenate((
        np.dot(rotation, np.asarray(X[:, :3]).T).T,
        substract_mean(np.dot(rotation, np.asarray(X[:, 3:6]).T).T),
        np.dot(rotation, np.asarray(X[:, 6:]).T).T), axis=1)
    return X


def check_label(Y, filter_threshold=0.0):
    """Check correctness of labels"""
    
    y_label = np.max(Y)
    if y_label >= 0:
        if (sum(Y == y_label) / Y.shape[0]) < filter_threshold:
            y_label = -1
    return y_label, label_is_valid(y_label)
    

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


def get_generator(train_patient, window_size, batch_size, temporal,
                  problem, validation_patient=None,
                  shift_augmentation=None,
                  rotate_augmentation=None, threshold=None,
                  settings=None):
    """"""

    global_std = get_data_property(problem=problem,
                                   property_key='std')
    if shift_augmentation is None:
        shift_augmentation = get_shift_augment()
    if rotate_augmentation is None:
        rotate_augmentation = get_rotate_augment()
    if threshold is None:
        threshold = get_filter_threshold()
    
    if settings is not None:
        settings = define_settings(
            settings, global_std=global_std,
            augment_shift=shift_augmentation,
            augment_rotate=rotate_augmentation, threshold=threshold)
        
    train_generator = generate_batches(
        train_patient, window_size, batch_size, temporal, global_std,
        shift_augmentation, rotate_augmentation, threshold)
    report_event('OK: Train Generator setup')
        
    if validation_patient is None:
        validation_generator = None
        report_event('WARNING: No validation data configured')
    else:
        validation_generator = generate_batches(
            validation_patient, window_size, batch_size, temporal,
            global_std, 0, 0, threshold)
        report_event('OK: Validation Generator setup')
        
    return train_generator, validation_generator, settings
    

if __name__ == '__main__':
    problem = 'fog'
    reproducibility = True
    seed = 77
    if reproducibility:
        np.random.seed(seed)
        rd.seed(seed)
    [train_data, val_data, test_data] = get_patient_split(
        problem=problem)

    full_batch_size = 128
    window_times = [2, 3]
    filter_thresholds = [0.5]
    data_freq = 100
    n_shift = 2
    n_rotate = 4
    for window_time in window_times:
        batch_size = int(round(full_batch_size / 2**(window_time-1)))
        window_size = int(window_time * data_freq)
        for filter_th in filter_thresholds:
            print('\nBatch_s:' + str(batch_size) + ' window:' +
                  str(window_time) + ' filter:' + str(filter_th))
            print('TRAIN')
            get_generator(
                train_data, window_size, batch_size, False,
                problem, validation_patient=None,
                shift_augmentation=2, rotate_augmentation=4,
                threshold=filter_th, settings=None)
            print('VAL')
            get_generator(
                val_data, window_size, batch_size, False,
                problem, validation_patient=None,
                shift_augmentation=0, rotate_augmentation=0,
                threshold=filter_th, settings=None)
            
    print('END')

# EOF

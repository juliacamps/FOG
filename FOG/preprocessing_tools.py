"""IO functions for FOG deep-learning project"""

# Authors: Julia Camps <julia.camps.sereix@est.fib.upc.edu>
# Created on: 06/10/2016 17:22

import numpy as np
import random as rd
from collections import Counter
from scipy.stats import truncnorm
# from itertools import combinations

from FOG.definitions import get_patient_partition
from FOG.definitions import get_data_structure
from FOG.definitions import define_settings
from FOG.definitions import get_inter_delimiter
from FOG.definitions import label_is_valid
from FOG.definitions import get_positive_label
from FOG.definitions import get_negative_label
from FOG.definitions import get_undefined_label
from FOG.definitions import get_file_duration

from FOG.utils import split_data
from FOG.utils import parse_value
from FOG.utils import fft_window

from FOG.experiment_conf import get_rotation_params

from FOG.io_functions import get_data_property
from FOG.io_functions import read_data_file


def generate_rotations(n_mat, roate_proba):
    """Return a random rotation matrix"""
    rotation_params = get_rotation_params()
    # X-Axis
    x_mean = rotation_params['x']['mean'] * np.pi / 180
    x_std = rotation_params['x']['std'] * np.pi / 180
    x_clip_a = rotation_params['x']['range'][0] * np.pi / 180
    x_clip_b = rotation_params['x']['range'][1] * np.pi / 180
    a, b = (x_clip_a - x_mean) / x_std, (x_clip_b - x_mean) / x_std
    xtheta = truncnorm(a=a, b=b, scale=x_std).rvs(size=n_mat) + x_mean
    # Y-Axis
    y_mean = rotation_params['y']['mean'] * np.pi / 180
    y_std = rotation_params['y']['std'] * np.pi / 180
    y_clip_a = rotation_params['y']['range'][0] * np.pi / 180
    y_clip_b = rotation_params['y']['range'][1] * np.pi / 180
    a, b = (y_clip_a - y_mean) / y_std, (y_clip_b - y_mean) / y_std
    ytheta = truncnorm(a=a, b=b, scale=y_std).rvs(size=n_mat) + y_mean
    # Z-Axis
    z_mean = rotation_params['z']['mean'] * np.pi / 180
    z_std = rotation_params['z']['std'] * np.pi / 180
    z_clip_a = rotation_params['z']['range'][0] * np.pi / 180
    z_clip_b = rotation_params['z']['range'][1] * np.pi / 180
    a, b = (z_clip_a - z_mean) / z_std, (z_clip_b - z_mean) / z_std
    ztheta = truncnorm(a=a, b=b, scale=z_std).rvs(size=n_mat) + z_mean

    R = []
    for i in range(n_mat):
        if rd.random() < roate_proba:
            Rx = np.asarray(
                [[1, 0, 0], [0, np.cos(xtheta[i]), -np.sin(xtheta[i])],
                 [0, np.sin(xtheta[i]), np.cos(xtheta[i])]])
            Ry = np.matrix([[np.cos(ytheta[i]), 0, np.sin(ytheta[i])],
                            [0, 1, 0],
                            [-np.sin(ytheta[i]), 0, np.cos(ytheta[i])]])
            Rz = np.matrix([[np.cos(ztheta[i]), -np.sin(ztheta[i]), 0],
                            [np.sin(ztheta[i]), np.cos(ztheta[i]), 0], [0, 0, 1]])
            R.append(np.dot(np.dot(Rx, Ry), Rz))
        else:
            R.append(np.identity(3))
    return R


def generate_batches(data, window_size, batch_size,
                     global_std, pure_threshold, stacking,
                     temporal=False, predict=False,
                     get_raw_data=False, single_file=False,
                     augmentation_factor=1, roate_proba=0.):
    """Create data set generator"""

    condition = True
    if predict:
        augmentation_factor = 1

    # Prepare data
    if single_file:
        epoch_data = [data]
    elif isinstance(data, dict):
        epoch_data = [file for patient in list(
            data.items()) for file in patient[1]]
    else:
        epoch_data = data
        
    n_files = len(epoch_data)

    if not (temporal or predict):
        file_size = len(epoch_data[0])
        iterators = []
        for augment_it in range(augmentation_factor):
            for file_it in range(n_files):
                for window_it in range(1, file_size, window_size):
                    iterators.append((augment_it, file_it,
                                      window_it))
    # Infinite loop
    while condition:
        # batch_count = 0
        # y_cum = 0
        condition = not single_file
        epoch_shifts = []
        epoch_rotate = []
        
        for augment_it in range(augmentation_factor):
            shift = []
            if predict:
                roate_proba = 0.
                for z in range(n_files):
                    shift.append(0)
            else:
                shift = [int(x) for x in np.around(
                    np.random.uniform(low=0.0, high=1.0,
                                      size=n_files) * window_size)]

            epoch_rotate.append(generate_rotations(n_files,
                                                   roate_proba))
            epoch_shifts.append(shift)
            
        if temporal or predict:
            for augment_it in range(augmentation_factor):
                rd.shuffle(epoch_data)
                rotation_matrices = epoch_rotate[augment_it]
                shifts = epoch_shifts[augment_it]
                
                for j in range(n_files):
                    file_data = epoch_data[j]
                    rotate_mat = rotation_matrices[j]
                    shift = shifts[j]

                    batch_spectral_X = []
                    batch_temporal_X = []
                    batch_pre_temporal_X = []
                    batch_Y = []
                    batch_it = 0
                    ant_X = None
                    ant_raw = None
        
                    for X_raw, y_raw, y_orig in get_windows(
                            file_data, window_size, shift):

                        y, valid = check_label(y_raw,
                                               pure_threshold)
                        X_raw = preprocess_data(X_raw, global_std,
                                                rotate_mat)
                        
                        act_X = apply_FFT(X_raw, stacking)
                        
                        if ant_X is None:
                            ant_X = act_X.copy()
                            ant_raw = X_raw.copy()
                        if valid and window_size == X_raw.shape[0]:
        
                            X_spectral = apply_stacking(act_X, ant_X, stacking)
                            # X_temporal = np.concatenate((ant_raw,X_raw), axis=0)
                            X_temporal = X_raw
                            pre_X_temporal = ant_raw
                            
                            batch_spectral_X.append(X_spectral)
                            batch_temporal_X.append(X_temporal)
                            batch_pre_temporal_X.append(
                                pre_X_temporal)
                            batch_Y.append(y)
                            batch_it += 1
    
                            if batch_it == batch_size:
                                yield ([np.asarray(
                                    batch_spectral_X), np.asarray(
                                    batch_temporal_X), np.asarray(batch_pre_temporal_X)],
                                           np.asarray(batch_Y))
                                
                                # y_cum_part = sum(np.asarray(
                                #     batch_Y) > 0)
                                # y_cum += y_cum_part
                                # batch_count += 1

                                batch_spectral_X = []
                                batch_temporal_X = []
                                batch_pre_temporal_X = []
                                batch_Y = []
                                batch_it = 0
        
                        ant_X = act_X.copy()
                        ant_raw = X_raw.copy()
        else:
            rd.shuffle(iterators)
            batch_spectral_X = []
            batch_temporal_X = []
            batch_pre_temporal_X = []
            batch_Y = []
            batch_it = 0
            for augment_it, file_it, window_it in iterators:
                rotate_mat = epoch_rotate[augment_it][file_it]
                shift = epoch_shifts[augment_it][file_it]

                data_ini = window_it - window_size + shift
                data_fi = data_ini + (window_size * 2)
                if data_fi > file_size:
                    continue
                X_raw, y_raw, y_orig = split_data(
                    epoch_data[file_it][data_ini:(data_fi), :])
                y_raw = y_raw[window_size:]
                
                y, valid = check_label(y_raw,
                                       pure_threshold)
                if valid:
                    
                    X_raw = preprocess_data(X_raw, global_std,
                                            rotate_mat)

                    # CAUTION!!! THE 2 FOLOWING LINES:
                    # Will only work if 'stacking' is greater than 0
                    # otherwise, it should be done in two steps at
                    # the cost of additional computation time
                    X = apply_FFT(X_raw, stacking*2)
                    X_spectral = apply_stacking(X[int(window_size/2):,:],
                                           X[:int(window_size/2),:],
                                           stacking)
                    X_temporal = X_raw[window_size:,:]
                    pre_X_temporal = X_raw[:window_size, :]

                    batch_spectral_X.append(X_spectral)
                    batch_temporal_X.append(X_temporal)
                    batch_pre_temporal_X.append(pre_X_temporal)
                    batch_Y.append(y)
                    batch_it += 1
        
                    if batch_it == batch_size:
                        yield ([np.asarray(batch_spectral_X),
                                np.asarray(batch_temporal_X), np.asarray(batch_pre_temporal_X)],
                               np.asarray(batch_Y))

                        # y_cum_part = sum(np.asarray(
                        #     batch_Y) > 0)
                        # y_cum += y_cum_part
                        # batch_count += 1
    
                        batch_spectral_X = []
                        batch_temporal_X = []
                        batch_pre_temporal_X = []
                        batch_Y = []
                        batch_it = 0

    #     n_samples = batch_count * batch_size
    #     if not predict:
    #         print('    '+"'"+'n_train'+"'"+':'+str(n_samples)
    #               +',\n'+"'"+'percent_pos_train'+"'"+': '+str(y_cum / n_samples) +',')
    #     else:
    #         print(''+"'"+'n_validation'+"'"+': '+str(n_samples)
    #               +',\n'+"'"+'percent_pos_val'+"'"+': '+str(y_cum / n_samples) +',')
    #     break
    # return 1


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


def get_patient_split(data_freq):
    """Split patients between train, validation and test groups
    
    Return
    ------
    train_patient, val_patient, test_patient : str array-like
    
    """
    [train, validation, test] = get_patient_partition()
    patient_data = get_data_structure(data_freq=data_freq)
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
    X = X / global_std
    X = np.concatenate((
            np.dot(rotation, np.asarray(X[:, :3]).T).T,
            np.dot(rotation, np.asarray(X[:, 3:]).T).T), axis=1)
    return X


def apply_FFT(X, stacking):
    """"""
    if stacking > 1:
        X_parts = np.split(X, stacking)
        X_new = fft_window(X_parts[0])
        for i in range(1, stacking, 1):
            X_new = np.concatenate((X_new, fft_window(X_parts[i])))
    elif stacking == 1:
        X_new = fft_window(X)
    else:
        X_new = X
    return X_new


def apply_stacking(act_X, ant_X, stacking):
    """"""
    if stacking > 1:
        act_X_parts = np.split(act_X, stacking)
        ant_X_parts = np.split(ant_X, stacking)
        X_new = np.concatenate((act_X_parts[0], ant_X_parts[-1]),
                               axis=1)
        for i in range(1, stacking, 1):
            new_part = np.concatenate((act_X_parts[i], act_X_parts[
                i-1]), axis=1)
            X_new = np.concatenate((X_new, new_part), axis=0)
    elif stacking == 1:
        X_new = np.concatenate((act_X, ant_X),  axis=1)
    else:
        X_new = act_X
    return X_new


def check_label(Y, filter_th=0.0):
    """Check correctness of labels"""
    y_count = Counter(Y)
    label = None
    if get_positive_label() in y_count:
        if (y_count[get_positive_label()] / Y.shape[0]) > filter_th:
            label = get_positive_label()
    elif get_undefined_label() in y_count:
        label = get_undefined_label()
    elif get_negative_label() in y_count:
        if (y_count[get_negative_label()] / Y.shape[0]) > filter_th:
            label = get_negative_label()
    if label is None:
        label = get_undefined_label()
    return label, label_is_valid(label)


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


def get_generator(train_data, window_size, batch_size,
                  stacking, pure_threshold, data_freq, n_feature,
                  validation_data=None,
                  settings=None, temporal=False, predict=False,
                  get_raw_data=False, single_file=False,
                  augmentation_factor=1, roate_proba=0.):
    """"""

    global_std = get_data_property(property_key='std',
                                   data_freq=data_freq, n_feature=n_feature)

    if settings is not None:
        settings = define_settings(
            settings, global_std=global_std,
            pure_threshold=pure_threshold)
    
    train_generator = generate_batches(
        train_data, window_size, batch_size, global_std,
        pure_threshold,
        stacking, temporal=temporal, predict=predict,
        get_raw_data=get_raw_data, single_file=single_file,
        augmentation_factor=augmentation_factor, roate_proba=roate_proba)
    # report_event('OK: Train Generator setup')
    
    if validation_data is None or single_file:
        validation_generator = None
        # report_event('WARNING: No validation data configured')
    else:
        # validation_data = get_dataset(validation_patient)
        validation_generator = generate_batches(
            validation_data, window_size, batch_size,
            global_std, pure_threshold=pure_threshold,
            stacking=stacking, temporal=temporal, predict=True)
        # report_event('OK: Validation Generator setup')
        
    return train_generator, validation_generator, settings
    

if __name__ == '__main__':
    reproducibility = False
    seed = 77
    if reproducibility:
        np.random.seed(seed)
        rd.seed(seed)

    data_freqs = [50]
    file_time = get_file_duration()

    n_feature = 6
    temporals = [True]
    configurations = [
        {
            'window_time': 2.56,
            'batch_size': 16,
            'pure_threshold': 0.5
        }
    ]
    for data_freq in data_freqs:
        [train_patient, val_patient,
         test_patient] = get_patient_split(data_freq)
        train_data = get_dataset(train_patient)
        validation_data = get_dataset(val_patient)
        test_data = get_dataset(test_patient)
        for temporal in temporals:
            for conf in configurations:
                window_time = conf['window_time']
                batch_size = conf['batch_size']
                pure_threshold = conf['pure_threshold']
                window_size = int(window_time * data_freq)
        
                print('('+"'"+'f'+str(data_freq)+'_w'+(str(
                    window_time).replace('.','_'))+'_b'+str(
                    batch_size)+'_p'+str(pure_threshold)+"'"
                      +', {\n    '+"'"+'data_freq'+"'"+': '
                      +str(data_freq)+',\n    '+"'"+'is_temporal'+"'"+': '
                      +str(temporal)+',\n    '+"'"+'window_time'+"'"+': '
                      +str(window_time)+',\n    '+"'"+'batch_size'+"'"+': '
                      +str(batch_size)+',\n    '+"'"+'pure_threshold'+"'"+': '
                      + str(pure_threshold)+',\n    '+"'"+'file_time'+"'"+': '
                      + str(file_time) +',')
                get_generator(
                    train_data, window_size, batch_size,
                    1, pure_threshold, data_freq, n_feature,
                    augmentation_factor=1,
                    validation_data=validation_data, temporal=temporal,
                    roate_proba=0.5)
                print(
                    '    ' + "'" + 'n_test' + "'" + ': ' +
                    str(0) + ',\n    ' + "'" +
                    'percent_pos_test' + "'" + ': ' + str(0)
                    + '\n}),')
                get_generator(
                    test_data, window_size, batch_size,
                    1, pure_threshold, data_freq, n_feature,
                    temporal=temporal, predict=True)
                
    print('END')

# EOF

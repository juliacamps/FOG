"""IO functions for FOG deep-learning project"""

# Authors: Julia Camps <julia.camps.sereix@est.fib.upc.edu>
# Created on: 06/10/2016 17:22

import numpy as np
import pandas as pd
import random as rd

from FOG.io_functions import get_dataset
from FOG.io_functions import save_matrix_data
from FOG.io_functions import get_all_patient
from FOG.io_functions import get_patient_data_files
from FOG.io_functions import is_correct
from FOG.io_functions import get_patient_names
from FOG.io_functions import get_std_mean

[_DATA_STD, _DATA_MEAN] = get_std_mean()
[_PATIENT_LIST, _VAL_PATIENT_DEFAULT, _TEST_PATIENT_DEFAULT] = \
    get_patient_names()
_SHIFT_RANGE = [0, 1]
_ROT_RANGE = np.array((-20, 20)) * np.pi / 180


def generate_arrays_from_file(path_list, window_size,
                              batch_size=32,
                              preprocess=False, temporal=False,
                              augment_count=[0],
                              augement_data_type='all',
                              filter_threshold=0.0):
    """Create data set generator"""
        
    shift_count = 1
    rotate_count = 1
    augment_data = False
    if len(augment_count) > 1:
        shift_count += augment_count[0]
        rotate_count += augment_count[1]
        augment_data = True
    elif augment_count[0] > 0:
        shift_count += augment_count[0]
        rotate_count += augment_count[0]
        augment_data = True

    while 1:
        batch_count = 0
        y_cum = 0
        aux_count = 0
        batch_X = []
        batch_Y = []
        batch_it = 0
        discarded = 0
        for path in path_list:
            # Data Augmentation
            shift_indexes = [0]
            rotate = [None]
            if augment_data:
                if (augement_data_type == 'all'
                        or augement_data_type == 'shift'):
                    shift_rates = [0.0]
                    for rand_shift in np.random.uniform(
                            _SHIFT_RANGE[0], _SHIFT_RANGE[1],
                            size=(shift_count - 1)):
                        shift_rates.append(rand_shift)
                    for shift in shift_rates:
                        shift_indexes.append(
                            int(round(window_size * shift)))
                        
                if (augement_data_type == 'all'
                        or augement_data_type == 'rotate'):
                    for i in range(rotate_count - 1):
                        rotate.append(random_rot())
                        
            for it_shift in range(shift_count):
                shift = shift_indexes[it_shift]
                for it_rot in range(rotate_count):
                    rotate_mat = rotate[it_rot]
                    if temporal:
                        batch_X = []
                        batch_Y = []
                        batch_it = 0
                    for X in pd.read_csv(
                            path, delim_whitespace=True,
                            dtype=float, chunksize=window_size,
                            na_filter=False, skiprows=shift):
                        X = (X.as_matrix())[:, :-1]
                        y = check_label(X[:, -1], filter_threshold)
                        if y == -1 and np.max(X[:, -1]) != -1:
                            discarded += 1
                        aux_count += X.shape[0]
                        if y >= 0 and window_size == X.shape[0]:
                            X = X[:, :-1]
                            if rotate_mat is not None:
                                X = np.concatenate((np.dot(rotate_mat,
                                    np.asarray(X[:, :3]).T).T,
                                    np.dot(rotate_mat, np.asarray(
                                        X[:, 3:6]).T).T, np.dot(
                                    rotate_mat, np.asarray(
                                        X[:, 6:]).T).T), axis=1)
                            batch_X.append(X)
                            batch_Y.append(y)
                            batch_it += 1
                            if batch_it == batch_size:
                                yield (np.asarray(batch_X),
                                       np.asarray(batch_Y))
                                y_cum += sum(np.asarray(batch_Y))
                                batch_count += 1
                                batch_X = []
                                batch_Y = []
                                batch_it = 0
                        elif temporal:
                            batch_it = 0
                            batch_X = []
                            batch_Y = []
                            # model_.reset_states()
            
            # model_.reset_states()
        # n_samples = batch_count * batch_size
        # print('Total number of samples in all data: ')
        # print(aux_count)
        # print('Num Batches: ' + str(batch_count))
        # print('Available number of samples: ' + str(
        #     n_samples))
        # print('Percentage of FOG:')
        # print(y_cum / n_samples)
        # print('Percentage of lost samples:')
        # print(1 - (n_samples * window_size) / aux_count)
        # print('Discarded for filtering: ' + str(discarded))
        # break


def confusion_matrix(y_true, y_pred):
    """"""
    conf_mat = np.zeros((2, 2))
    for i in range(np.asarray(y_true).shape[0]):
        if int(y_true[i]) == 1:
            if int(y_true[i]) == int(np.round(y_pred[i])):
                conf_mat[0, 0] += 1
            else:
                conf_mat[0, 1] += 1
        else:
            if int(y_true[i]) == int(np.round(y_pred[i])):
                conf_mat[1, 1] += 1
            else:
                conf_mat[1, 0] += 1
    return conf_mat


def load_instance(line, preprocess=False):
    """Transform raw input to sample data"""
    x = line[:-1]
    y = line[-1]
    if preprocess:
        x = check_data(x)
    return [x, y]


def split_data(patient_list, part=[0.7, 0.15, 0.15], random_=False):
    """Split batches between train, validation and test
    
    Parameter
    ---------
    path_list : str array-like
    part : float array-like, optional, default: [0.7, 0.15, 0.15]
        Percent of the overall data set corresponding to train,
        validation and test partitions, respectively.
    random_ : bool
        Indicates if the test patients should be random, due to
        reproducibility.
        
    Return
    ------
    test_patient, train_patient : str array-like
    
    """

    if random_:
        n_val = int(np.round(len(patient_list) * part[1]))
        n_test = int(np.round(len(patient_list) * part[2]))
        if n_val == 0 and part[1] > 0:
            n_val = 1
        if n_test == 0 and part[2] > 0:
            n_test = 1
        rd.shuffle(patient_list)
        test_patient = patient_list[:n_test]
        val_patient = patient_list[n_test:(n_val + n_test)]
        train_patient = patient_list[(n_val + n_test):]
    else:
        val_patient = _VAL_PATIENT_DEFAULT
        test_patient = _TEST_PATIENT_DEFAULT
        train_patient = [x for x in patient_list if x not
                         in test_patient and x not in val_patient]
    return [train_patient, val_patient, test_patient]


def check_data(X):
    """Center and normalize data.
    
    Parameters
    ----------
    X : array-like, shape=[n_sample, n_features]
    """
    X -= _DATA_MEAN
    X /= _DATA_STD
    return X


def check_label(Y, filter_threshold=0.0):
    """Check correctness of labels"""
    
    y_label = np.max(Y)
    if y_label >= 0:
        if (sum(Y == y_label) / Y.shape[0]) < filter_threshold:
            y_label = -1
    return y_label


def full_preprocessing(train_patient, type_name='all'):
    """Pre-calculates the pre-processing of all the training data
    
    Parameters
    ----------
    train_patient : str array-like
        List of training patients.
    type_name : str {'all', 'fog' or 'walk'}, optional, default: 'all'
        Indicate the objective data to be searched, problem wisely.
        
    """
    for patient in train_patient:
        train_file = get_patient_data_files(patient,
                                            type_name=type_name)
        for file in train_file:
            [X, y] = get_dataset(file)
            X = check_data(X)
            save_matrix_data(np.concatenate((X, y), axis=1), file)


def generate_dataset(part=[0.7, 0.15, 0.15]):
    """Generate dataset with partitions
    
    Parameters
    ----------
    test : float, optional, default: 0.20
        Percent of the overall data set corresponding to test data.
    
    Return
    ------
    test_patient, train_patient : str array-like
    
    """
    
    patient_list = get_all_patient()
    return split_data(patient_list, part=part, random_=False)
    

def std_mean(seq):
    """Calculate the sample-std and the mean
    
    Parameters
    ----------
    seq : array-like
    
    Return
    ------
    std_cum : float
        Sample standard deviation.
    mean_cum : float
        Mean.
    """
    it_n = 1
    mean_cum = 0
    std_cum = 0
    n_sample = len(seq)
    for i in range(n_sample):
        mean_diff = seq[i] - mean_cum
        mean_cum += (mean_diff / it_n)
        std_cum += (mean_diff * (seq[i] - mean_cum)) / (n_sample-1)
        it_n += 1
    std_cum = np.sqrt(std_cum)
    return [std_cum, mean_cum]


def random_rot():
    """Return a random rotation matrix"""
    theta = rd.uniform(_ROT_RANGE[0], _ROT_RANGE[1])
    Rx = np.matrix([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)]])
    theta = rd.uniform(_ROT_RANGE[0], _ROT_RANGE[1])
    Ry = np.matrix([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)]])
    theta = rd.uniform(_ROT_RANGE[0], _ROT_RANGE[1])
    Rz = np.matrix([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    R = np.dot(np.dot(Rx, Ry), Rz)
    return R


class AuxModel():
    """"""
    def reset_states(self):
        """"""


if __name__ == '__main__':
    problem = 'fog'
    reproducibility = True
    seed = 77
    if reproducibility:
        np.random.seed(seed)
        rd.seed(seed)
    patient_list = _PATIENT_LIST
    train_data = [patient for patient in patient_list
                  if (patient not in _VAL_PATIENT_DEFAULT
                      and patient not in _TEST_PATIENT_DEFAULT)]
    val_data = _VAL_PATIENT_DEFAULT
    test_data = _TEST_PATIENT_DEFAULT
    model = AuxModel()
    train_file = [file for patient in train_data for file in
                  get_patient_data_files(patient,
                                         type_name=problem)]
    val_file = [file for patient in val_data for file in
                  get_patient_data_files(patient,
                                         type_name=problem)]
    test_file = [file for patient in test_data for file in
                get_patient_data_files(patient,
                                       type_name=problem)]
    batch_sizes = [32]
    time_windows = [3]
    filter_thresholds = [0.5]
    data_freq = 100
    n_shift = 2
    n_rotate = 4
    for batch_it in range(len(batch_sizes)):
        batch_size = batch_sizes[batch_it]
        
        time_window = time_windows[batch_it]
        window_size = int(time_window * data_freq)
        for filter_th in filter_thresholds:
            print('\nBatch_s:' + str(batch_size) + ' window:' +
                  str(time_window) + ' filter:' + str(filter_th))
            print('TRAIN')
            generate_arrays_from_file(train_file, window_size,
                                      batch_size=batch_size,
                                      augment_count=[n_shift, n_rotate],
                                      augement_data_type='all',
                                      filter_threshold=filter_th)
            print('VALIDATION')
            generate_arrays_from_file(val_file, window_size,
                                      batch_size=batch_size,
                                      filter_threshold=filter_th)
            print('TEST')
            generate_arrays_from_file(test_file, window_size,
                                      batch_size=batch_size,
                                      filter_threshold=filter_th)
    print('END')

# EOF

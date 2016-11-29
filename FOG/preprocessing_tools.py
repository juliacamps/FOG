"""IO functions for FOG deep-learning project"""

# Authors: Julia Camps <julia.camps.sereix@est.fib.upc.edu>
# Created on: 06/10/2016 17:22

import numpy as np
import pandas as pd
import random as rd

from collections import Counter

from FOG.io_functions import get_dataset
from FOG.io_functions import get_std_mean
from FOG.io_functions import get_patient_data_files
from FOG.utils import substract_mean
from FOG.definitions import get_patient_groups
from FOG.definitions import get_patient
from FOG.definitions import get_rotate_range


_GLOBAL_STD = get_std_mean()[0]
_ROT_RANGE = get_rotate_range()


def generate_arrays_from_file(path_list, window_size=200,
                              batch_size=64,
                              preprocess=False, temporal=False,
                              augment_count=[0],
                              filter_threshold=0.0,
                              normalize=False):
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
        batch_Y_orig = []
        batch_it = 0
        discarded = 0
        for path in path_list:
            # Data Augmentation
            shift_indexes = [0]
            rotate = [np.identity(3)]
            if augment_data:
                if shift_count > 1:
                    for rand_shift in np.random.uniform(
                            low=0.0, high=1.0,
                            size=(shift_count - 1)):
                        shift_indexes.append(
                            int(round(window_size * rand_shift)))
                        
                if rotate_count > 1:
                    for i in range(rotate_count - 1):
                        rotate.append(random_rot())
                        
            for it_shift in range(shift_count):
                shift = shift_indexes[it_shift]
                for it_rot in range(rotate_count):
                    rotate_mat = rotate[it_rot]
                    if temporal:
                        batch_X = []
                        batch_Y = []
                        batch_Y_orig = []
                        batch_it = 0
                    for X in pd.read_csv(
                            path, delim_whitespace=True,
                            dtype=float, chunksize=window_size,
                            na_filter=False, skiprows=shift,
                            header=None):
                        y_orig = (X.as_matrix())[:, -1]
                        X = (X.as_matrix())[:, :-1]
                        y = check_label(X[:, -1], filter_threshold)
                        if y == -1 and np.max(X[:, -1]) != -1:
                            discarded += 1
                        aux_count += X.shape[0]
                        if y >= 0 and window_size == X.shape[0]:
                            X = preprocess_data(X[:, :-1], rotate_mat)
                            
                            batch_X.append(X)
                            batch_Y.append(y)
                            batch_Y_orig.append(y_orig)
                            batch_it += 1
                            if batch_it == batch_size:
                                yield (np.asarray(batch_X),
                                       np.asarray(batch_Y),
                                       np.asarray(batch_Y_orig))
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
        # print('Discarded only for threshold filtering: '
        #       + str(discarded))
        # break


def check_orig(y_orig_raw):
    """"""
    return Counter(y_orig_raw)


def split_data(patient_list=get_patient(), part=[0.7, 0.15, 0.15],
               random_=False):
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
        [train_patient, val_patient, test_patient] \
            = get_patient_groups()
    return [train_patient, val_patient, test_patient]


def preprocess_data(X, rotation=np.identity(3)):
    """Center and normalize data.
    
    Parameters
    ----------
    X : array-like, shape=[n_sample, n_features]
    """
    X /= _DATA_STD
    X = np.concatenate((
        np.dot(rotation, np.asarray(X[:, :3]).T).T,
        substract_mean(np.dot(rotation, np.asarray(X[:, 3:6]).T).T),
        np.dot(rotation, np.asarray(X[:, 6:]).T).T), axis=1)
    
    return X


def extract_original_labels(X):
    """"""
    return [X[:, :-1], np.asarray(X[:, -1])]


def check_label(Y, filter_threshold=0.0):
    """Check correctness of labels"""
    
    y_label = np.max(Y)
    if y_label >= 0:
        if (sum(Y == y_label) / Y.shape[0]) < filter_threshold:
            y_label = -1
    return y_label


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


def random_rot(range=_ROT_RANGE):
    """Return a random rotation matrix"""
    theta = rd.uniform(-range[0], range[0])
    Rx = np.matrix([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)]])
    theta = rd.uniform(-range[1], range[1])
    Ry = np.matrix([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)]])
    theta = rd.uniform(-range[2], range[2])
    Rz = np.matrix([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    R = np.dot(np.dot(Rx, Ry), Rz)
    return R


def get_train_validation_path(train_patient,
                              validation_patient=None, problem='fog'):
    """"""
    train_file = [file for patient in train_patient for file in
                  get_patient_data_files(patient, problem=problem)]
    if validation_patient is not None:
        validation_file = [file for patient in validation_patient for
                           file in get_patient_data_files(patient,
                                                          problem
                                                          =problem)]
    else:
        validation_file = None
    return [train_file, validation_file]


def get_generator(train_patient, validation_patient=None,
                  problem='fog', augment_shift=0, augment_rotate=0,
                  threshold=0.0, normalize=False, window_size=200,
                  batch_size=64):
    """"""
    [train_file, validation_file] = get_train_validation_path(
        train_patient, validation_patient, problem=problem)

    train_generator = generate_arrays_from_file(
        train_file, window_size=window_size, batch_size=batch_size,
        augment_count=[augment_shift, augment_rotate],
        filter_threshold=threshold, normalize=normalize)

    if validation_file is not None:
        validation_generator = generate_arrays_from_file(
            validation_file, window_size, batch_size=batch_size,
            filter_threshold=threshold, normalize=normalize)
    else:
        validation_generator = None
        
    return [train_generator, validation_generator]
    

class AuxModel:
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
                                         problem=problem)]
    val_file = [file for patient in val_data for file in
                  get_patient_data_files(patient,
                                         problem=problem)]
    test_file = [file for patient in test_data for file in
                get_patient_data_files(patient,
                                       problem=problem)]

    special_test_file = ['/home/juli/PycharmProjects/Keras_Projects'
                         '/data/fsl11/walk_fsl11_110.csv']

    full_batch_size = 128
    window_times = [2]
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
            print('SPECIAL_TEST')
            generate_arrays_from_file(special_test_file, window_size,
                                      batch_size=batch_size)
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

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


_DATA_STD = [27.8379884006, 34.0541798945, 20.2561325897,
             3.4419919171, 3.4928714944, 3.7259256299, 0.2658480797,
             0.2957936948, 0.3050834012]
_DATA_MEAN = [-22.3235421779, -14.2307838391, 9.1812591243,
              -1.9621648871, -7.9875374392, -0.7413856581,
              0.2767370913, 0.7317886181, 0.1967207557]

_VAL_PATIENT_DEFAULT = ['mac20', 'tek12', 'fsl13']
_TEST_PATIENT_DEFAULT = ['fsl18', 'mac10', 'tek24', 'nui13', 'tek23']
_DATA_AUGMENTATION = ['shift', 'noise']
_SHIFT_RANGE = [0, 1]
_ROT_RANGE = np.array((-20, 20)) * np.pi / 180
_REPRODUCIBILITY = True
_SEED = 277
_N_FEATURES = 9


def generate_arrays_from_file(model_, path_list, window_size,
                              batch_size=32,
                              preprocess=False, temporal=False,
                              augment_count=[0],
                              augement_data_type='shift'):
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
            
                    batch_X = []
                    batch_Y = []
                    batch_it = 0
                    
                    for X in pd.read_csv(
                            path, delim_whitespace=True,
                            dtype=float, chunksize=window_size,
                            na_filter=False, skiprows=shift):
    
                        X = X.as_matrix()
                        y = np.max(X[:, -1])
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
                        else:
                            batch_it = 0
                            batch_X = []
                            batch_Y = []
                            # model_.reset_states()
            
            # model_.reset_states()
        # n_samples = batch_count * batch_size
        # print('Total number of samples in all data: ')
        # print(aux_count)
        # print('Num Batches: ' + str(batch_count))
        # print('Theoretical number of samples: ' + str(
        #     n_samples))
        # print('Percentage of FOG:')
        # print(y_cum / n_samples)
        # print('Percentage of lost samples:')
        # print(1 - (n_samples * window_size) / aux_count)
        # break


def load_instance(line, preprocess=False):
    """Transform raw input to sample data"""
    x = line[:-1]
    y = line[-1]
    if preprocess:
        x = check_data(x)
    return [x, y]


def process_line(line):
    """Process line"""
    
    return [line[:-1], line[-1]]


def split_data(patient_list, test=0.15, random_=False,
               validation=False):
    """Split batches between train and test/validation
    
    Parameter
    ---------
    path_list : str array-like
    test : float, optional, default: 0.15
        Percent of the overall data set corresponding to test data.
    random_ : bool
        Indicates if the test patients should be random, due to
        reproducibility.
        
    Return
    ------
    test_patient, train_patient : str array-like
    
    """

    if random_:
        n_test = int(np.round(len(patient_list) * test))
        if n_test == 0 and test > 0:
            n_test = 1
        rd.shuffle(patient_list)
        test_patient = patient_list[:n_test]
        train_patient = patient_list[n_test:]
    elif validation:
        test_patient = _VAL_PATIENT_DEFAULT
        train_patient = [x for x in patient_list if x not
                         in test_patient]
    else:
        test_patient = _TEST_PATIENT_DEFAULT
        train_patient = [x for x in patient_list if x not
                         in test_patient]
    return [test_patient, train_patient]


def check_data(X):
    """Center and normalize data.
    
    Parameters
    ----------
    X : array-like, shape=[n_sample, n_features]
    
    """
    
    X -= _DATA_MEAN
    X /= _DATA_STD
    return X


def check_label(Y):
    """Check correctness of labels"""
    
    y_label = -1
    for y in Y:
        if y >= 0:
            y_label = max((y_label, y))
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
            save_matrix_data(
                np.concatenate((X, np.reshape(y, (y.shape[0], 1))),
                               axis=1), file
            )


def generate_dataset(test=0.2):
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
    return split_data(patient_list, test=test)
    

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
    
    reproducibility = True
    seed = 277
    if reproducibility:
        np.random.seed(seed)
        rd.seed(seed)
    
    patient_list = ['nui16', 'tek07', 'mac03', 'tek04', 'tek24',
                    'mac17', 'mac21', 'tek25', 'mac04', 'mac07',
                    'tek12','fsl11', 'mac12', 'mac19', 'tek23',
                    'fsl18', 'fsl14', 'nui13', 'fsl24', 'fsl20',
                    'fsl16', 'fsl15', 'fsl17', 'mac10', 'fsl13',
                    'nui14', 'nui06', 'mac20', 'nui01']
    
    train_data = [patient for patient in patient_list
                  if (patient not in _VAL_PATIENT_DEFAULT
                      and patient not in _TEST_PATIENT_DEFAULT and
                      is_correct(patient))]
    val_data = _VAL_PATIENT_DEFAULT
    test_data = _TEST_PATIENT_DEFAULT
    model = AuxModel()
    train_file = [file for patient in train_data for file in
                  get_patient_data_files(patient,
                                         type_name='walk')]
    val_file = [file for patient in val_data for file in
                  get_patient_data_files(patient,
                                         type_name='walk')]
    test_file = [file for patient in test_data for file in
                get_patient_data_files(patient,
                                       type_name='walk')]

    batch_size = 64
    data_freq = 40
    n_shift = 3
    n_rotate = 6
    time_window = 1
    window_size = int(time_window * data_freq)

    generate_arrays_from_file(model, train_file, window_size,
                              batch_size=batch_size,
                              augment_count=[n_shift, n_rotate],
                              augement_data_type='all')
    
    print('END1')
    generate_arrays_from_file(model, val_file, window_size,
                              batch_size=batch_size)
    print('END2')
    generate_arrays_from_file(model, test_file, window_size,
                              batch_size=batch_size)
    print('END3')
    # print('END')


# EOF

"""IO functions for FOG deep-learning project"""

# Authors: Julia Camps <julia.camps.sereix@est.fib.upc.edu>
# Created on: 06/10/2016 17:22
from xml.sax import default_parser_list

import numpy as np
import pandas as pd

from random import shuffle

from FOG.io_functions import get_dataset
from FOG.io_functions import save_matrix_data
from FOG.io_functions import get_all_patient
from FOG.io_functions import get_patient_data_files


_DATA_STD = [27.8457523169, 34.0549218804, 20.2572144113,
             3.4417279114, 3.49333072, 3.7251159781, 0.265846121,
             0.2957929075, 0.3050630071]
_DATA_MEAN = [-22.3248296633, -14.23055157, 9.182059959,
              -1.9622041635, -7.9877904591, -0.7412694559,
              0.2767348722, 0.7317887307, 0.196745521]

_VAL_PATIENT_DEFAULT = ['mac20', 'tek12', 'fsl13']
_TEST_PATIENT_DEFAULT = ['fsl18', 'mac17', 'tek24', 'nui13', 'tek23']
_DATA_AUGMENTATION = ['shift', 'noise']
_SHIFT_RANGE = [-0.25, 0.25]
_REPRODUCIBILITY = True
_SEED = 177
_N_FEATURES = 9


def generate_arrays_from_file(model_, path_list, window_size,
                              window_spacing, batch_size=32,
                              preprocess=False, temporal=False,
                              augment_count=0,
                              augement_data_type='shift'):
    """Create data set generator"""
    augment_data = (augment_count > 0)
    shift_indexes = [0]
    noise = [np.zeros((window_size, _N_FEATURES))]
    if augment_data:
        if _REPRODUCIBILITY:
            np.random.seed(_SEED)
        if (augement_data_type == 'all'
                or augement_data_type == 'shift'):
            shift_rates = [0.0]
            for rand_shift in np.random.uniform(_SHIFT_RANGE[0],
                                                _SHIFT_RANGE[1],
                                                size=augment_count):
                shift_rates.append(rand_shift)
            for shift in shift_rates:
                if shift < 0:
                    shift = 1 - shift
                shift_indexes.append(int(round(window_size * shift)))
        if (augement_data_type == 'all'
                or augement_data_type == 'noise'):
            for i in range(augment_count):
                noise.append(np.random.normal(0, 1, (window_size,
                                                     _N_FEATURES)))
    state = False
    while 1:
        batch_count = 0
        y_cum = 0
        aux_count = 0
        for path in path_list:
            for it_augment in range(augment_count + 1):
                gaussian_noise = noise[it_augment]
                shift = shift_indexes[it_augment]
        
                X_old = None
                batch_X = []
                batch_Y = []
                batch_it = 0
                for X_raw in pd.read_csv(path, delim_whitespace=True,
                                     dtype=float, chunksize=window_spacing,
                                     na_filter=False, skiprows=shift):
                    aux_count += len(X_raw)
                    X_raw = X_raw.as_matrix()
                    y = np.max(X_raw[:, -1])
                    
                    if (y >= 0 and window_spacing == X_raw.shape[0]):
                        X_raw = X_raw[:, :-1]
                        if X_old is not None:
                            X = (np.concatenate((np.asarray(X_old),
                                                X_raw))
                                 + gaussian_noise)
                            
                            batch_X.append(X)
                            batch_Y.append(y)
                            batch_it += 1
                            if batch_it == batch_size:
                                # yield (np.asarray(batch_X).reshape(
                                #     (batch_size, window_size,
                                #      _N_FEATURES, 1)),
                                #        np.asarray(batch_Y).reshape(
                                #            (batch_size, 1)))
                                y_cum += sum(np.asarray(batch_Y))
                                batch_count += 1
                                batch_X = []
                                batch_Y = []
                                batch_it = 0
                        X_old = X_raw
                    else:
                        # if batch_it > 0:
                            # yield (np.asarray(batch_X).reshape(
                            #     (batch_size, window_size,
                            #      _N_FEATURES, 1)),
                            #        np.asarray(batch_Y).reshape(
                            #            (batch_size, 1)))
                            # batch_count += batch_it / batch_size
                        X_old = None
                        batch_it = 0
                        batch_X = []
                        batch_Y = []
                        # model_.reset_states()
     
            
            # model_.reset_states()
        n_samples = batch_count * batch_size
        print('Total number of samples in all data: ')
        print(aux_count)
        print('Num Batches: ' + str(batch_count))
        print('Theoretical number of samples: ' + str(
            n_samples))
        print('Percentage of FOG:')
        print(y_cum / n_samples)
        print('Percentage of lost samples:')
        print(1 - (n_samples * window_spacing) / aux_count)
        break

                
                        # if correct:
                        #     # create numpy arrays of input data
                        #     # and labels, from each line in the file
                        #     X = []
                        #     Y = []
                        #     for it_seq in range(window_spacing):
                        #         # if (it_line + it_seq) >= len(lines):
                        #             # print(it_line)
                        #             # print(it_seq)
                        #             # print(window_spacing)
                        #             # print(len(lines))
                        #             # exit(1)
                        #         [x, y] = load_instance(
                        #             lines[it_line + it_seq],
                        #             preprocess)
                        #         X.append(x)
                        #         Y.append(y)
                        #     # [X, Y] = [load_instance(lines[it_line
                        #     #                               + it_seq],
                        #     #                         preprocess)
                        #     #           ]
                        #     X = np.array(X)
                        #     # print(len(X[0]))
                        #     Y = np.array(Y)
                            # print(X.shape)
                            # print(gaussian_noise_1.shape)
                            # if gaussian_it:
                            #     X += gaussian_noise_1
                            # else:
                            #     X += gaussian_noise_2
                            # gaussian_it = not gaussian_it
                            # y_new = check_label(Y)
                            # # y_label = max(y_new, y_old)
                            # y_label = y_new
                            # y_old = y_new
                            # if X_old is not None:
                            #     if y_label >= 0:
                            #         sample_count += 1
                            #         if temporal:
                            #             state = True
                            #         X_data = np.concatenate(
                            #             (X, X_old))
                            #         X_data = X_data.reshape((
                            #             X_data.shape[0],
                            #             X_data.shape[1], 1))
                            #         batch_data.append(X_data)
                            #         batch_label.append(y_label)
                            #         batch_count += 1
                            #         total_sample_count += 1
                            #         if batch_count == batch_size:
                                        # batch_X = np.array(batch_data)
                                        # batch_Y = np.array(
                                        #     batch_label).reshape((
                                        #     batch_size, 1))
                                        # print(len(batch_data))
                                        # print(len(batch_label))
                                        # print(len(batch_data[0]))
                                        # y_train = batch_Y.reshape(
                                        #     (-1, 1))
                                        # print('Sample number:')
                                        # print(total_batch_count)
                                        # print('Labels:')
                                        # print(batch_label)
                                        # y_cum += sum(batch_label)
                            #             print(np.asarray(
                            #                 batch_data),
                            #                    np.array(batch_label
                            #                             ).reshape(
                            #                        (-1, 1)))
                            #             exit(1)
                            #             batch_count = 0
                            #             batch_data = []
                            #             batch_label = []
                            #             total_batch_count += 1
                            #     elif y_label == -1:
                            #
                            #         X = None
                            #         X_old = None
                            #         y_old = -1
                            #         # if state:
                            #             # model_.reset_states()
                            #     else:
                            #         print('Error with labels with '
                            #               'value: ' + str(y_label))
                            # if X is not None:
                            #     X_old = X.copy()
                            #     y_old = y_label
            # if state:
                # model_.reset_states()
        # print('Finished reading all data')
        #
        # print('Samples number: ' + str(sample_count))
        # if state:
            # model_.reset_states()
        # print('Total number of samples in all data: ')
        # print(total_sample_count)
        # print('Num Batches: ' + str(total_batch_count))
        # print('Theoretical number of samples: ' + str(
        #     total_batch_count * batch_size))
        # print('Percentage of FOG:')
        # print(y_cum/(total_batch_count * batch_size))
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
        shuffle(patient_list)
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
        
 
def gen_k_folds(patient_list, n_fold=10):
    """Generate k-fold partitions
    
    Parameters
    ----------
    patient_list : str array-like
        List of patients corresponding to the training data set.
    n_fold : int, optional, default: 10
        Number of partitions to be done.
    
    Return
    ------
    folds : str array-like, shape=[n_fold, (n_patient/n_fold)]
        List of patients per each fold.
        
    """
    n_patient = len(patient_list)
    fold = []
    patient_fold_up = np.ceil(n_patient / n_fold)
    patient_fold_down = np.floor(n_patient / n_fold)
    shuffle(patient_list)
    it_ = 0
    for i in range(n_fold):
        if ((n_patient - (i * patient_fold_up)) / n_fold >
                patient_fold_down):
            fold.append(patient_list[it_: (it_ + patient_fold_up)])
            it_ += patient_fold_up
        else:
            fold.append(patient_list[it_: (it_ + patient_fold_down)])
            it_ += patient_fold_down
    return fold


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


class AuxModel():
    """"""
    def reset_states(self):
        """"""

if __name__ == '__main__':
    patient_list = ['nui16', 'tek07', 'mac03', 'tek04', 'tek24',
                    'mac17', 'mac21', 'tek25', 'mac04', 'mac07',
                    'tek12','fsl11', 'mac12', 'mac19', 'tek23',
                    'fsl18', 'fsl14', 'nui13', 'fsl24', 'fsl20',
                    'fsl16', 'fsl15', 'fsl17', 'mac10', 'fsl13',
                    'nui14', 'nui06', 'mac20', 'nui01']
    train_data = [patient for patient in patient_list
                  if (patient not in _VAL_PATIENT_DEFAULT
                      and patient not in _TEST_PATIENT_DEFAULT)]
    val_data = _VAL_PATIENT_DEFAULT
    test_data = _TEST_PATIENT_DEFAULT
    model = AuxModel()
    train_file = [file for patient in train_data for file in
                  get_patient_data_files(patient,
                                         type_name='fog')]
    val_file = [file for patient in val_data for file in
                  get_patient_data_files(patient,
                                         type_name='fog')]
    test_file = [file for patient in test_data for file in
                get_patient_data_files(patient,
                                       type_name='fog')]
    # print(train_file)
    batch_size = 50
    data_freq = 200
    time_window = 1
    window_overlaping = 0.5
    window_size = int(time_window * data_freq)
    window_spacing = int(round(window_size * (1 - window_overlaping)))

    # print('Start')
    generate_arrays_from_file(model, train_file,
                                        window_size,
                                                window_spacing,
                                                batch_size=batch_size,
                                                augment_count=0)
    
    print('END1')
    generate_arrays_from_file(model, val_file, window_size,
                              window_spacing,
                              batch_size=batch_size,
                              augment_count=0)
    print('END2')
    generate_arrays_from_file(model, test_file, window_size,
                              window_spacing,
                              batch_size=batch_size,
                              augment_count=0)
    print('END3')
    # print('END')


# EOF

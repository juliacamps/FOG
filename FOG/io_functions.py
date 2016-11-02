"""IO functions for FOG deep-learning project"""

# Authors: Julia Camps <julia.camps.sereix@est.fib.upc.edu>
# Created on: 06/10/2016 17:22

import errno
import numpy as np
import scipy.io as sio

from os import listdir
from os import remove
from os import rmdir
from os import mkdir
from os import getcwd
from os.path import join
from os.path import isdir
from os.path import isfile
from itertools import groupby
from operator import itemgetter

from keras.models import model_from_json

_WORKING_DIR = getcwd()
_DEFAULT_DATA_PATH = _WORKING_DIR + '/../../data/'
_DEFAULT_RAW_DATA_PATH = _WORKING_DIR + '/../../RAW_DATA/'
_DEFAULT_MODEL_PATH = _WORKING_DIR + '/../../model/'
_DEFAULT_DATA_EXT = '.csv'
_DEFAULT_RAW_DATA_EXT = '.MAT'
_DEFAULT_MODEL_EXT = '.h5'


def silent_remove(path):
    """Remove file or dir"""
    if isfile(path):
        _silent_remove_file(path)
    elif isdir(path):
        for part_path in listdir(path):
            silent_remove(join(path, part_path))
        _silent_remove_dir(path)


def _silent_remove_dir(dir_path):
    """Remove directory"""
    try:
        rmdir(dir_path)
    # This would be "except OSError, e:" before Python 2.6
    except OSError as e:
        # errno.ENOENT = no such file or directory
        if e.errno != errno.ENOENT:
            # re-raise exception if a different error occurred
            raise


def _silent_remove_file(file_path):
    """Remove file"""
    try:
        remove(file_path)
    # This would be "except OSError, e:" before Python 2.6
    except OSError as e:
        # errno.ENOENT = no such file or directory
        if e.errno != errno.ENOENT:
            # re-raise exception if a different error occurred
            raise


def get_all_patient(data_path=_DEFAULT_DATA_PATH):
    """Get list of patient
    
    Parameters
    ----------
    data_path : str, optional, default: content of _DEFAULT_DATA_PATH
        Indicate the path containing the patients directories.
            
    Return
    ------
    dataset_path : str array-like, shape=[n_patient,]
    
    """

    dataset_path = [f for f in listdir(data_path) if isdir(join(
        data_path, f))]
    return dataset_path


def get_patient_data_files(patient, type_name='all',
                           data_path=_DEFAULT_DATA_PATH):
    """Get data of a patient

    Parameters
    ----------
    patient : str
        ID of a patient in the data set. Should match with the
        folder, containing its data, name.
    type_name : str {'all', 'fog' or 'walk'}, optional, default: 'all'
        Indicate the objective data to be searched, problem wisely.
    data_path : str, optional, default: content of _DEFAULT_DATA_PATH
        Indicate the path containing the patients directories.

    Return
    ------
    patient_dataset : str array-like, shape=[n_files_per_patient,]
        Paths of patient files, corresponding the type of data
        defined.
    """
    
    if type_name == 'all':
        ini_name = ''
    else:
        ini_name = type_name
    patient_path = data_path + patient + '/'
    patient_dataset = [join(patient_path, f) for f in
                       listdir(patient_path) if isfile(
        join(patient_path, f)) and f.startswith(ini_name)]
    return patient_dataset


def get_dataset(data_file_path, label_pos=9):
    """Get data in path indicated
    
    Parameters
    ----------
    data_file_path : str
        Indicate the path of a data file.
    label_pos : int
        Column index corresponding to the labels value.
        
    Return
    ------
    X : float matrix, shape=[n_samples, n_features-1]
        Data instances in file.
    y : float array-like, shape=[n_samples,]
        Labels of the instances in X.
        
    """
    return read_data(data_file_path, label_pos=label_pos)
    
    
def get_data_path(type_name='', data_path=_DEFAULT_DATA_PATH,
                  ext=_DEFAULT_DATA_EXT, group=False):
    """Get data paths
    
    Parameters
    ------------
    type_name : str {'all', 'fog' or 'walk'}, optional, default: 'all'
        Name of the data set that is being searched.
    data_path : str, optional, default: '../../../data/'
        Data sets path (or relative path).
    ext : str {'.csv' or '.MAT'}, optional, default: '.csv'
        Files extension.
    group : bool, optional, default: False
        Activate the feature of pairing file-names patient wisely.
        
    Return
    ------
    dataset_path : str array-like, shape=[n_files,]
    
    """

    if group:
        dataset_path_sparse = [[[join(data_path, f), f[:f.find('.')]],
                                f[: (f.rfind('_'))]] for f in
                               listdir(data_path) if
                               isfile(join(data_path, f)) and
                               f.endswith(ext) and
                               f.startswith(type_name)]
        sorted_input = sorted(dataset_path_sparse, key=itemgetter(1))
        file_group = groupby(sorted_input, key=itemgetter(1))
        dataset_path = {k: [x[0] for x in v] for k, v in file_group}
    else:
        dataset_path = [[join(data_path, f), f[:f.find('.')]] for f in
                        listdir(data_path)
                        if isfile(join(data_path, f)) and f.endswith(
                ext) and f.startswith(type_name)]

    return dataset_path


def get_raw_data_path(data_path=_DEFAULT_RAW_DATA_PATH,
                      ext=_DEFAULT_RAW_DATA_EXT, group=False):
    """Get raw data paths

        Parameters
        ------------
        data_path : str, optional, default: '../../../RAW_DATA/'
            Data sets path (or relative path).
        ext : str {'.csv' or '.MAT'}, optional, default: '.MAT'
            Files extension.
        group : bool, optional, default: False
            Activate the feature of pairing file-names patient wisely.

        Return
        ------
        dataset_path : str array-like, shape=[n_files,]

        """
    return get_data_path(data_path=data_path, ext=ext, group=group)


def save_model(model_, name):
    """Save keras model"""
    model_json = model_.to_json()
    with open(_DEFAULT_MODEL_PATH + name + '.json', 'w') as json_file:
        json_file.write(model_json)
    model_.save(_DEFAULT_MODEL_PATH + name + _DEFAULT_MODEL_EXT)
    

def load_model(name):
    """Load model weights"""
    # load json and create model
    with open(_DEFAULT_MODEL_PATH + name + '.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(_DEFAULT_MODEL_PATH + name +
                               _DEFAULT_MODEL_EXT)
    print("Loaded model from disk")
    return loaded_model


def save_matrix_data(X, file_name, data_path=_DEFAULT_DATA_PATH,
                     ext=_DEFAULT_DATA_EXT, delimiter=' ',
                     print_=False):
    """Get raw data paths

    Parameters
    ----------
    X : float matrix
        Data to be saved.
    file_name : str
        Name for the current data to be saved.
    data_path : str, optional, default: '../../../RAW_DATA/'
        Data sets path (or relative path).
    ext : str {'.csv' or '.MAT'}, optional, default: '.MAT'
        Files extension.
    delimiter : str, optional, default: ' '
        Specifies the columns separation character, in the saved file.
    print_ : bool
        Indicate special saving case for string data matrix.

    """
    
    if isfile(file_name):
        file_path = file_name
    else:
        file_path = data_path + file_name + ext
    silent_remove(file_path)
    if print_:
        with open(file_path, 'w') as f:
            for row in X:
                print(row)
                line = ' '.join(row)
                print(line, file=f)
    else:
        np.savetxt(file_path, X, delimiter=delimiter)
    

def read_data(path_name, label_pos=None):
    """Read data from path
    
    Parameters
    ----------
    path_name : str
    label_pos : int array-like, optional, default: None
        Indexes composing the labels of the data set.

    Return
    ------
    X : float matrix, shape=[n_samples, n_features]
        Data instances.
    y : int, shape=[n_samples,]
        Labels of the instances.
    """
    X = None
    y = None

    if path_name.endswith(_DEFAULT_DATA_EXT):
        X = np.loadtxt(path_name)
    elif path_name.endswith(_DEFAULT_RAW_DATA_EXT):
        X = sio.loadmat(path_name)
    if label_pos is not None and X is not None:
        y = X[:, label_pos]
        X = np.delete(X, label_pos, axis=1)
    return [X, y]


def prepare_path(patient_name):
    """Prepare the file system for the new patients data.
    
    Parameters
    ----------
    patient_name : str
        ID of the new patient data.
    
    Return
    ------
    new_path : str
    """
    new_path = _DEFAULT_DATA_PATH + patient_name + '/'
    silent_remove(new_path)
    mkdir(new_path)
    return new_path

# EOF

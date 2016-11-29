"""Utils functions"""

# Authors: Julia Camps <julia.camps.sereix@est.fib.upc.edu>
# Created on: 29/11/2016 11:26

import numpy as np
from os import getcwd


# FILESYSTEM PROPERTIES
_WORKING_DIR = getcwd()
_DEFAULT_DATA_PATH = _WORKING_DIR + '/../../data/'
_DEFAULT_RAW_DATA_PATH = _WORKING_DIR + '/../../RAW_DATA/'
_DEFAULT_MODELS_PATH = _WORKING_DIR + '/../model/'
_DEFAULT_DATA_EXT = '.csv'
_DEFAULT_RAW_DATA_EXT = '.MAT'
_DEFAULT_MODEL_EXT = '.h5'

# META-DATA INFORMATION
_CORRECT_PATIENTS = ['fsl11', 'fsl13', 'fsl14', 'fsl15', 'fsl17',
                     'fsl18', 'fsl20', 'mac03', 'mac04', 'mac07',
                     'mac10', 'mac12', 'mac20', 'nui01', 'nui13',
                     'tek04', 'tek07', 'tek12', 'tek23', 'tek24',
                     'tek25']
_PATIENT_LIST = ['nui16', 'tek07', 'mac03', 'tek04', 'tek24',
                 'mac17', 'mac21', 'tek25', 'mac04', 'mac07',
                 'tek12', 'fsl11', 'mac12', 'mac19', 'tek23',
                 'fsl18', 'fsl14', 'nui13', 'fsl24', 'fsl20',
                 'fsl16', 'fsl15', 'fsl17', 'mac10', 'fsl13',
                 'nui14', 'nui06', 'mac20', 'nui01']
_TEST_PATIENT_DEFAULT = ['mac20', 'tek12', 'fsl13']
_VAL_PATIENT_DEFAULT = ['fsl18', 'mac10', 'tek24', 'nui13', 'tek23']
_ACTIVITY_CLASS_KEY = [0, 3, 5, 6, 7, 8, 11, 12, 13, 20, 22, 23, 28,
                       29, 30, 31, 32]

# CONFIGURATION KEYWORDS FOR I/O TRAINED MODELS
_EXPERIMENT_CONFIGURATION = ['model_conf', 'window_time',
                             'window_size', 'batch_size', 'n_epoch',
                             'n_parameter', 'dropout',
                             'regularization', 'atrous', 'pooling',
                             'initialization', 'threshold',
                             'n_train', 'percent_pos_train',
                             'n_validation', 'percent_pos_val',
                             'augment_shift', 'augment_rotate',
                             'random_seed', 'normalize', 'problem',
                             'train_patient', 'val_patient',
                             'final_status'
                             ]
_PREDICTION_KEY = [['train', ['global_result',
                              'partial_result']],
                   ['validation', ['global_result',
                                   'partial_result']]
                   ]
_DELIMITER = ' '

# METRICS BEING CONSIDERED
_EXISTING_METRIC = {'conf_mat': np.zeros((2, 2)), 'accuracy': 0,
                    'sensitivity': 0, 'specificity': 0}

# AUGMENTATION SETTINGS
_ROT_RANGE = np.array((30, 45, 10)) * np.pi / 180


def get_rotate_range():
    """"""
    return _ROT_RANGE


def get_data_path():
    """"""
    return _DEFAULT_DATA_PATH


def get_raw_data_path():
    """"""
    return _DEFAULT_RAW_DATA_PATH


def get_data_ext():
    """"""
    return _DEFAULT_DATA_EXT


def get_raw_data_ext():
    """"""
    return _DEFAULT_RAW_DATA_EXT


def get_models_path():
    """"""
    return _DEFAULT_MODELS_PATH


def get_models_ext():
    """"""
    return _DEFAULT_MODEL_EXT


def get_patient():
    """Get list of patient"""
    return _CORRECT_PATIENTS


def get_patient_groups():
    """"""
    return [[patient for patient in _CORRECT_PATIENTS if patient
            not in _VAL_PATIENT_DEFAULT
             and patient not in _TEST_PATIENT_DEFAULT],
            _VAL_PATIENT_DEFAULT, _TEST_PATIENT_DEFAULT]


def get_configuration_key():
    """"""
    return _EXPERIMENT_CONFIGURATION


def get_prediction_key():
    """"""
    return _PREDICTION_KEY


def get_line_key():
    """"""
    return [['EXPERIMENT_INFORMATION', ['date', 'model_name',
                                        'conf_name']],
            ['EXPERIMENT_CONFIGURATION', get_configuration_key()],
            ['TRAIN', ['train_acc', 'train_sensitivity',
                       'train_specificity', 'training_time']],
            ['VALIDATION', ['validation_acc',
                            'validation_sensitivity',
                            'validation_specificity']],
            ['TRAIN_FULL_LOG', ['train_log']]]


def is_correct_patient(patient_name):
    """Check if patient is in the correct patients list"""
    return patient_name in _CORRECT_PATIENTS


def get_delimiter():
    """"""
    return _DELIMITER


def get_activity_class():
    """"""
    return _ACTIVITY_CLASS_KEY


def get_existing_metric():
    """"""
    return [metric_key for metric_key, metric_init in
            _EXISTING_METRIC]


def get_metric_init(metric_key):
    """"""
    init = None
    if metric_key in _EXISTING_METRIC:
        init = _EXISTING_METRIC[metric_key]
    return init

# EOF

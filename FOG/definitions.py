"""Utils functions"""

# Authors: Julia Camps <julia.camps.sereix@est.fib.upc.edu>
# Created on: 29/11/2016 11:26

import numpy as np
from os import getcwd
from os import listdir
from os.path import isdir
from os.path import isfile
from os.path import join
from itertools import groupby
from operator import itemgetter

from FOG.experiment_conf import _get_log_mode_id
from FOG.io_functions import report_event


# FILESYSTEM PROPERTIES
_WORKING_DIR = getcwd()
_DEFAULT_DATA_PATH = _WORKING_DIR + '/../../data/'
_DEFAULT_RAW_DATA_PATH = _WORKING_DIR + '/../../RAW_DATA/'
_DEFAULT_MODELS_PATH = _WORKING_DIR + '/../model/'
_DEFAULT_DATA_EXT = '.csv'
_DEFAULT_RAW_DATA_EXT = '.MAT'
_DEFAULT_MODEL_WEIGHT_EXT = '.h5'
_DEFAULT_MODEL_STRUCTURE_EXT = '.json'
_DEFAULT_MODEL_LOG_EXT = '.csv'
_DEFAULT_EVALUATION_NAME = 'evaluation'
_DEFAULT_EVALUATION_EXT = '.csv'
_DEFAULT_DATA_SUMMARY_FILE = 'info'
_DEFAULT_DATA_SUMMARY_PATH = _DEFAULT_DATA_PATH \
                             + _DEFAULT_DATA_SUMMARY_FILE
_DEFAULT_LOG_FILE_PATH = _WORKING_DIR + 'log_file.log'

# MODEL FILE KEYS
_MODEL_LOG_KEY = 'log'
_MODEL_WEIGHT_KEY = 'weight'
_MODEL_STRUCTURE_KEY = 'structure'

# META-DATA INFORMATION
_CORRECT_PATIENTS = ['fsl11', 'fsl13', 'fsl14', 'fsl15', 'fsl17',
                     'fsl18', 'fsl20', 'mac03', 'mac04', 'mac07',
                     'mac10', 'mac12', 'mac20', 'nui01', 'nui13',
                     'tek04', 'tek07', 'tek12', 'tek23', 'tek24',
                     'tek25'
                     ]
_PATIENT_LIST = ['nui16', 'tek07', 'mac03', 'tek04', 'tek24',
                 'mac17', 'mac21', 'tek25', 'mac04', 'mac07',
                 'tek12', 'fsl11', 'mac12', 'mac19', 'tek23',
                 'fsl18', 'fsl14', 'nui13', 'fsl24', 'fsl20',
                 'fsl16', 'fsl15', 'fsl17', 'mac10', 'fsl13',
                 'nui14', 'nui06', 'mac20', 'nui01'
                 ]
_TEST_PATIENT_DEFAULT = ['mac20', 'tek12', 'fsl13']
_VAL_PATIENT_DEFAULT = ['fsl18', 'mac10', 'tek24', 'nui13', 'tek23']
_ACTIVITY_CLASS_KEY = [0, 3, 5, 6, 7, 8, 11, 12, 13, 20, 22, 23, 28,
                       29, 30, 31, 32]

# CONFIGURATION KEYWORDS FOR I/O TRAINED MODELS
_CONFIGURATION_KEY = ['model_name', 'model_conf', 'window_time',
                      'window_size', 'batch_size', 'n_epoch',
                      'n_parameter', 'dropout', 'regularization',
                      'atrous', 'pooling', 'initialization',
                      'threshold', 'n_train', 'percent_pos_train',
                      'n_validation', 'percent_pos_val',
                      'augment_shift', 'augment_rotate',
                      'random_seed', 'problem',
                      'train_patient', 'val_patient', 'final_status'
                      ]
_PREDICTION_GLOBAL_KEY = 'global_result'
_PREDICTION_PARTIAL_KEY = 'partial_result'
_PREDICTION_KEY = [['train', [_PREDICTION_GLOBAL_KEY,
                              _PREDICTION_PARTIAL_KEY]],
                   ['validation', [_PREDICTION_GLOBAL_KEY,
                                   _PREDICTION_PARTIAL_KEY]]
                   ]
_LOG_SECTION_KEY = 'train_log'
_LOG_INFO_STRUCTURE = [
    ['EXPERIMENT_INFORMATION', ['date', 'conf_name']],
    ['EXPERIMENT_CONFIGURATION', _CONFIGURATION_KEY],
    ['TRAIN', ['train_acc', 'train_sensitivity',
               'train_specificity', 'training_time']],
    ['VALIDATION', ['validation_acc', 'validation_sensitivity',
                    'validation_specificity']],
    ['TRAIN_FULL_LOG', [_LOG_SECTION_KEY]]]

# EVALUATION
_EVALUATION_HEADERS = ['model_name', 'final_status',
                       'percent_pos_train',
                       'percent_pos_val', 'train_acc',
                       'train_sensitivity', 'train_specificity',
                       'training_time', 'validation_acc',
                       'validation_sensitivity',
                       'validation_specificity',
                       'window_time',
                       'window_size', 'batch_size', 'n_epoch',
                       'threshold', 'model_conf']

# CODING ASSUMPTIONS
_DELIMITER = ' '
_INNER_DELIMITER = '_'
_MODEL_PREFIX = 'model'
_CORRECT_STATUS = 'OK'
_LOG_MODES = {0: 'verbose', 1: 'errors', 2: 'quiet'}

# METRICS BEING CONSIDERED
_CONF_MAT_KEY = 'conf_mat'
_METRICS_TO_CALCULATE_KEY = ['accuracy', 'sensitivity', 'specificity']
_EXISTING_METRIC = {_CONF_MAT_KEY: np.zeros((2, 2)),
                    _METRICS_TO_CALCULATE_KEY[0]: 0,
                    _METRICS_TO_CALCULATE_KEY[1]: 0,
                    _METRICS_TO_CALCULATE_KEY[2]: 0}


# PROTECTED FUNCTIONS: ENCAPSULATE DIRECT DEFINITIONS (e.g. PATHS)
# Therefore, should be never accessed by functions from other modules
#   GLOBAL DEFINITIONS


def _valid_log_mode(log_mode_id):
    """"""
    return log_mode_id in _LOG_MODES


def _get_log_mode(log_mode_id):
    """"""
    return _LOG_MODES[log_mode_id]


def is_verbose_log_mode():
    """"""
    log_mode_id = _get_log_mode_id()
    verbose = False
    if _valid_log_mode(log_mode_id):
        if _get_log_mode(log_mode_id) == 'verbose':
            verbose = True
    return verbose


def is_errors_log_mode():
    """"""
    log_mode_id = _get_log_mode_id()
    errors = False
    if _valid_log_mode(log_mode_id):
        if _get_log_mode(log_mode_id) == 'errors':
            errors = True
    return errors


def is_quiet_log_mode():
    """"""
    log_mode_id = _get_log_mode_id()
    quiet = False
    if _valid_log_mode(log_mode_id):
        if _get_log_mode(log_mode_id) == 'quiet':
            quiet = True
    return quiet


def _get_delimiter():
    """"""
    return _DELIMITER


def _get_inner_delimiter():
    """"""
    return _INNER_DELIMITER


def to_string(data):
    """"""
    if isinstance(data, (list, tuple, np.ndarray)):
        data_str = _get_delimiter().join([to_string(data_part) for
                                          data_part in data])
    else:
        try:
            data_str = str(data)
        except Exception as e:
            msg = 'ERROR: STRING_CONVERSION_ERROR: ' + str(repr(e))
            report_event(msg, is_error=True)
            data_str = str(repr(e)).replace(_get_delimiter(),
                                            _get_inner_delimiter())
        else:
            data_str = data_str.replace(_get_delimiter(),
                                        _get_inner_delimiter())
    return data_str


def get_status_ini():
    """"""
    return _CORRECT_STATUS


def check_status(status):
    """"""
    return status == _CORRECT_STATUS


def get_log_file():
    """"""
    return _DEFAULT_LOG_FILE_PATH


#   FILESYSTEM: DATA PATHS
def _get_data_path(raw_data=False):
    """"""
    if raw_data:
        path = _DEFAULT_RAW_DATA_PATH
    else:
        path = _DEFAULT_DATA_PATH
    return path


def _get_data_ext(raw_data=False):
    """"""
    if raw_data:
        ext = _DEFAULT_RAW_DATA_EXT
    else:
        ext = _DEFAULT_DATA_EXT
    return ext


def _is_raw_data_path(path):
    """"""
    return path.endswith(_get_data_ext(raw_data=True))


def _is_processed_data_path(path):
    """"""
    return path.endswith(_get_data_ext(raw_data=False))


def check_data_path(path):
    """"""
    return [isfile(path) and (_is_raw_data_path(path)
                              or _is_processed_data_path(path)),
            _is_raw_data_path(path)]


def get_data_paths(problem, path=None, raw_data=False):
    """Get data paths

    Parameters
    ------------
    problem : str {'all', 'fog' or 'walk'}, optional, default: 'all'
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
    if path is None:
        path = _get_data_path(raw_data=raw_data)
    ext = _get_data_ext(raw_data=raw_data)
    prefix = get_prefix(problem=problem, raw_data=raw_data)
    
    dataset_path_sparse = [[[join(path, f), f[:f.find('.')]],
                            f[:(f.rfind('_'))]] for f in
                           listdir(path) if
                           isfile(join(path, f)) and
                           f.endswith(ext) and
                           f.startswith(prefix)]
    sorted_input = sorted(dataset_path_sparse, key=itemgetter(1))
    file_group = groupby(sorted_input, key=itemgetter(1))
    dataset_path = {k: [x[0] for x in v] for k, v in file_group}
    
    return dataset_path


def get_patient_path(patient_name, patients_path=None):
    """"""
    if patients_path is None:
        patients_path = _get_data_path()
    return patients_path + patient_name + '/'


#   FILESYSTEM: MODELS
def get_model_log_key():
    """"""
    return _MODEL_LOG_KEY


def get_model_weight_key():
    """"""
    return _MODEL_WEIGHT_KEY


def get_model_structure_key():
    """"""
    return _MODEL_STRUCTURE_KEY


def _get_models_path():
    """"""
    return _DEFAULT_MODELS_PATH


def _get_models_weight_ext():
    """"""
    return _DEFAULT_MODEL_WEIGHT_EXT


def _get_models_structure_ext():
    """"""
    return _DEFAULT_MODEL_STRUCTURE_EXT


def _get_models_log_ext():
    """"""
    return _DEFAULT_MODEL_LOG_EXT


def _get_models_prefix(problem):
    """"""
    # TODO: Do something different if the problem is fog or walk
    return _MODEL_PREFIX


def get_model_path(model_name, models_path=None):
    """"""
    if models_path is None:
        models_path = _get_models_path()
    return models_path + model_name + '/'


def get_models_paths(problem, models_path=None):
    """"""
    if models_path is None:
        models_path = _get_models_path()
    return [get_model_path(model_name=model_name,
                           models_path=models_path) for model_name
            in listdir(models_path) if  model_name.startswith(
            get_prefix(problem=problem, is_model=True))]


def get_model_out_file_path(model_name, model_path=None):
    """"""
    if model_path is None:
        model_path = get_model_path(model_name)
    

def _get_next_model_id(problem, models_path=None):
    """"""
    # TODO: Do something different if the problem is fog or walk
    if models_path is None:
        models_path = _get_models_path()
    next_id = -1
    for model_file in listdir(models_path):
        old_id = int(model_file[model_file.find('_') + 1])
        next_id = max(next_id, old_id)
    return next_id + 1


def get_new_model_name(problem, models_path=None):
    """"""
    if models_path is None:
        models_path = _get_models_path()
    return (get_prefix(problem=problem, is_model=True)
            + str(_get_next_model_id(models_path)))


#   FILESYSTEM: GENERAL
def get_prefix(problem, raw_data=False, is_model=False):
    """"""
    if is_model:
        prefix = (_get_models_prefix(problem=problem)
                  + _get_inner_delimiter())
    elif raw_data or problem == 'all':
        prefix = ''
    else:
        prefix = problem + _get_inner_delimiter()
    return prefix


def get_evaluation_headers():
    """"""
    return _EVALUATION_HEADERS


def get_evaluation_file(problem):
    """"""
    # TODO: Do something different if the problem is fog or walk
    return _DEFAULT_EVALUATION_NAME + _DEFAULT_EVALUATION_EXT


#   DATA FILES NAMES AND OTHER META-DATA
def get_patient():
    """Get list of patient"""
    return _CORRECT_PATIENTS


def get_patient_partitions():
    """"""
    return [[patient for patient in _CORRECT_PATIENTS if patient
            not in _VAL_PATIENT_DEFAULT
             and patient not in _TEST_PATIENT_DEFAULT],
            _VAL_PATIENT_DEFAULT, _TEST_PATIENT_DEFAULT]


def get_activity_class():
    """"""
    return _ACTIVITY_CLASS_KEY


#   METRICS KEYS
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


def get_metric_to_calculate_key():
    """"""
    return _METRICS_TO_CALCULATE_KEY


def get_conf_mat_key():
    """"""
    return _CONF_MAT_KEY


#   CONFIGURATION PROPERTIES KEYS
def get_settings_log_key():
    """"""
    return _LOG_SECTION_KEY
    
    
def get_prediction_global_key():
    """"""
    return _PREDICTION_GLOBAL_KEY


def get_prediction_partial_key():
    """"""
    return _PREDICTION_PARTIAL_KEY


def get_configuration_key():
    """"""
    return _CONFIGURATION_KEY


def get_prediction_key():
    """"""
    return _PREDICTION_KEY


def get_line_key():
    """"""
    return _LOG_INFO_STRUCTURE

# EOF

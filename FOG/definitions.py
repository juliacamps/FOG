"""Utils functions"""

# Authors: Julia Camps <julia.camps.sereix@est.fib.upc.edu>
# Created on: 29/11/2016 11:26

from os import getcwd
from os import listdir
from os.path import isdir
from os.path import isfile
from os.path import join
from collections import OrderedDict


# CODING DEFINITIONS
_OUTER_DELIMITER = ' '
_INNER_DELIMITER = '_'
_DELIMITER = ['\n', _OUTER_DELIMITER]
_MODEL_PREFIX = 'model'
_CORRECT_STATUS = 'OK'
_LOG_MODES = {0: 'verbose', 1: 'errors', 2: 'quiet'}
_LABEL_VALUES = {'undefined': -1, 'negative': 0, 'positive': 1}

# FILESYSTEM PROPERTIES
_WORKING_DIR = getcwd()
_DEFAULT_DATA_PATH = _WORKING_DIR + '/../../data/'
_DEFAULT_RAW_DATA_PATH = _WORKING_DIR + '/../../RAW_DATA/'
_DEFAULT_MODELS_PATH = _WORKING_DIR + '/../model/'
_DEFAULT_DATA_EXT = '.csv'
_DEFAULT_RAW_DATA_EXT = '.MAT'
_DEFAULT_SUMMARY_PATH = _DEFAULT_MODELS_PATH
_DEFAULT_SUMMARY_FILE_NAME = 'summary.csv'
_DEFAULT_DATA_INFO_FILE = 'info.csv'
_DEFAULT_LOG_FILE_PATH = _WORKING_DIR + 'detailed_log_file.log'
_DEFAULT_RUN_LOG_FILE_PATH = _WORKING_DIR + 'general_log_file.log'
_PATH_WORD_DELIMITER = '_'

# MODEL FILES
_MODEL_SETTINGS_FILE_KEY = 0
_MODEL_WEIGHT_FILE_KEY = 1
_MODEL_STRUCTURE_FILE_KEY = 2
_MODEL_FILE_NAME = {_MODEL_SETTINGS_FILE_KEY: 'log',
                    _MODEL_WEIGHT_FILE_KEY: 'weight',
                    _MODEL_STRUCTURE_FILE_KEY: 'structure'}
_MODEL_FILE_EXT = {_MODEL_SETTINGS_FILE_KEY: '.csv',
                   _MODEL_WEIGHT_FILE_KEY: '.h5',
                   _MODEL_STRUCTURE_FILE_KEY: '.json'}

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
# METRICS BEING CONSIDERED
_METRIC_KEYS = ['conf_mat', 'accuracy', 'sensitivity', 'specificity']
_TRAIN_KEY = 'train'
_VAL_KEY = 'validation'
_TEST_KEY = 'test'
_TRAIN_RESULT_KEY = [_TRAIN_KEY + _INNER_DELIMITER + metric for
                     metric in _METRIC_KEYS]
_VAL_RESULT_KEY = [_VAL_KEY + _INNER_DELIMITER + metric for metric
                   in _METRIC_KEYS]
_TEST_RESULT_KEY = [_TEST_KEY + _INNER_DELIMITER + metric for
                    metric in _METRIC_KEYS]
# CONFIGURATION KEYWORDS FOR I/O TRAINED MODELS
_CONFIGURATION_KEY = (['model_name', 'date', 'final_status',
                       'n_train', 'percent_pos_train',
                       'n_validation', 'percent_pos_val', 'n_test',
                       'percent_pos_test', 'training_time']
                      + _TRAIN_RESULT_KEY + _VAL_RESULT_KEY
                      + ['window_time', 'window_size', 'batch_size',
                         'n_epoch', 'model_conf', 'n_parameter',
                         'threshold', 'augment_shift',
                         'augment_rotate', 'random_seed', 'problem',
                         'train_patient', 'val_patient',
                         'reproducibility', 'temporal', 'global_std'
                         ])
_PREDICTION_GLOBAL_KEY = 'global_result'
_PREDICTION_PARTIAL_KEY = 'class_result'

_PREDICTION_KEY = [[_TRAIN_KEY, [_PREDICTION_GLOBAL_KEY,
                                 _PREDICTION_PARTIAL_KEY]],
                   [_VAL_KEY, [_PREDICTION_GLOBAL_KEY,
                               _PREDICTION_PARTIAL_KEY]]
                   ]


# EVALUATION
_SUMMARY_HEADERS = ['date', 'model_name', 'final_status',
                    'percent_pos_train', 'percent_pos_val',
                    'training_time',
                    ] + _TRAIN_RESULT_KEY + _VAL_RESULT_KEY + [
                    'window_time', 'window_size', 'batch_size',
                    'n_epoch', 'model_conf']


# PROTECTED FUNCTIONS: ENCAPSULATE DIRECT DEFINITIONS (e.g. PATHS)
# Therefore, should be never accessed by functions from other modules
#   SETTINGS DEFINITIONS
def generate_summary(settings, summary_metrics):
    """"""
    summary = OrderedDict([(key, None) for key in _SUMMARY_HEADERS])
    for key, val in summary.items():
        if key in settings:
            summary[key] = settings[key]
        elif key in summary_metrics:
            summary[key] = summary_metrics[key]
    return summary


def define_settings(settings, new_settings_dict=None,
                    model_name=None, date=None,
                    final_status=None, n_train=None,
                    percent_pos_train=None, n_validation=None,
                    percent_pos_val=None, n_test=None,
                    percent_pos_test=None, train_accuracy=None,
                    train_sensitivity=None,
                    train_specificity=None, training_time=None,
                    validation_accuracy=None,
                    validation_sensitivity=None,
                    validation_specificity=None, test_accuracy=None,
                    test_sensitivity=None,
                    test_specificity=None, window_time=None,
                    window_size=None, batch_size=None, n_epoch=None,
                    model_conf=None, n_parameter=None, threshold=None,
                    augment_shift=None, augment_rotate=None,
                    random_seed=None, problem=None,
                    train_patient=None, val_patient=None,
                    reproducibility=None, temporal=None,
                    global_std=None):
    """"""
    if new_settings_dict is not None:
        for setting_key, setting_value in new_settings_dict.items():
            if setting_key in settings:
                settings[setting_key] = setting_value
    if model_name is not None:
        settings['model_name'] = model_name
    if date is not None:
        settings['date'] = date
    if final_status is not None:
        settings['final_status'] = final_status
    if n_train is not None:
        settings['n_train'] = n_train
    if percent_pos_train is not None:
        settings['percent_pos_train'] = percent_pos_train
    if n_validation is not None:
        settings['n_validation'] = n_validation
    if percent_pos_val is not None:
        settings['percent_pos_val'] = percent_pos_val
    if n_test is not None:
        settings['n_test'] = n_test
    if percent_pos_test is not None:
        settings['percent_pos_test'] = percent_pos_test
    if train_accuracy is not None:
        settings['train_accuracy'] = train_accuracy
    if train_sensitivity is not None:
        settings['train_sensitivity'] = train_sensitivity
    if train_specificity is not None:
        settings['train_specificity'] = train_specificity
    if training_time is not None:
        settings['training_time'] = training_time
    if validation_accuracy is not None:
        settings['validation_accuracy'] = validation_accuracy
    if validation_sensitivity is not None:
        settings['validation_sensitivity'] = validation_sensitivity
    if validation_specificity is not None:
        settings['validation_specificity'] = validation_specificity
    if test_accuracy is not None:
        settings['test_accuracy'] = test_accuracy
    if test_sensitivity is not None:
        settings['test_sensitivity'] = test_sensitivity
    if test_specificity is not None:
        settings['test_specificity'] = test_specificity
    if window_time is not None:
        settings['window_time'] = window_time
    if window_size is not None:
        settings['window_size'] = window_size
    if batch_size is not None:
        settings['batch_size'] = batch_size
    if n_epoch is not None:
        settings['n_epoch'] = n_epoch
    if model_conf is not None:
        settings['model_conf'] = model_conf
    if n_parameter is not None:
        settings['n_parameter'] = n_parameter
    if threshold is not None:
        settings['threshold'] = threshold
    if augment_shift is not None:
        settings['augment_shift'] = augment_shift
    if augment_rotate is not None:
        settings['augment_rotate'] = augment_rotate
    if random_seed is not None:
        settings['random_seed'] = random_seed
    if problem is not None:
        settings['problem'] = problem
    if train_patient is not None:
        settings['train_patient'] = train_patient
    if val_patient is not None:
        settings['val_patient'] = val_patient
    if reproducibility is not None:
        settings['reproducibility'] = reproducibility
    if temporal is not None:
        settings['temporal'] = temporal
    if global_std is not None:
        settings['global_std'] = global_std
    return settings
    
    
def init_settings():
    """"""
    return OrderedDict([(conf_key, None) for
                        conf_key in _CONFIGURATION_KEY])


def get_prediction_summary(train_metric, val_metric=None):
    """"""
    summary = OrderedDict([(key, None)
                           for key in (_TRAIN_RESULT_KEY
                                       + _VAL_RESULT_KEY)])
    for metric_name, metric_value in train_metric.items():
        train_metric_key = _TRAIN_KEY + _INNER_DELIMITER + metric_name
        if train_metric_key in summary:
            summary[train_metric_key] = metric_value
    if val_metric is not None:
        for metric_name, metric_value in val_metric.items():
            val_metric_key = _VAL_KEY + _INNER_DELIMITER + metric_name
            if val_metric_key in summary:
                summary[val_metric_key] = metric_value
    return summary


#   LABEL VALUES DEFINITIONS
def label_is_valid(label_value):
    """"""
    return (label_value == _LABEL_VALUES['negative']
            or label_value == _LABEL_VALUES['positive'])


def label_is_positive(label_value):
    """"""
    return label_value == _LABEL_VALUES['positive']


def label_is_negative(label_value):
    """"""
    return label_value == _LABEL_VALUES['negative']


#   LOG MODES
def _valid_log_mode(log_mode_id):
    """"""
    return log_mode_id in _LOG_MODES


def _get_log_mode(log_mode_id):
    """"""
    return _LOG_MODES[log_mode_id]


def is_verbose_log_mode(log_mode_id):
    """"""
    verbose = False
    if _valid_log_mode(log_mode_id):
        if _get_log_mode(log_mode_id) == 'verbose':
            verbose = True
    return verbose


def is_errors_log_mode(log_mode_id):
    """"""
    errors = False
    if _valid_log_mode(log_mode_id):
        if _get_log_mode(log_mode_id) == 'errors':
            errors = True
    return errors


def is_quiet_log_mode(log_mode_id):
    """"""
    quiet = False
    if _valid_log_mode(log_mode_id):
        if _get_log_mode(log_mode_id) == 'quiet':
            quiet = True
    return quiet


#   GLOBAL DELIMITER DEFINITIONS
def get_delimiter_level(n_level=None):
    """"""
    if n_level is None:
        n_level = len(_DELIMITER)
    return _DELIMITER[:n_level]


def get_inter_delimiter():
    """"""
    return _OUTER_DELIMITER


def get_delimiter():
    """"""
    return [_DELIMITER, _OUTER_DELIMITER, _INNER_DELIMITER]


#   STATUS DEFINITIONS AND INTERPRETATION
def get_status_ini():
    """"""
    return _CORRECT_STATUS


def check_status(status):
    """"""
    return status == _CORRECT_STATUS


#   FILESYSTEM: LOG FILES
def get_log_file():
    """"""
    return _DEFAULT_LOG_FILE_PATH


def get_run_log_file():
    """"""
    return _DEFAULT_RUN_LOG_FILE_PATH
    

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


def get_data_structure(problem, data_path=None, raw_data=False):
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
    if data_path is None:
        data_path = _get_data_path(raw_data=raw_data)
    data_ext = _get_data_ext(raw_data=raw_data)
    files_prefix = get_prefix(problem=problem, raw_data=raw_data)
    if raw_data:
        data_structure = _get_raw_data_structure(
            data_path, data_ext, files_prefix)
    else:
        data_structure = _get_parsed_data_structure(
            data_path, data_ext, files_prefix)
    return data_structure


def _get_raw_data_structure(data_path, data_ext, files_prefix):
    """"""
    patient = _get_all_patient()
    data_structure = {}
    for file_name in listdir(data_path):
        file_path = join(data_path, file_name)
        patient_name = file_name[: (file_name.rfind(
            _PATH_WORD_DELIMITER))]
        if (isfile(file_path) and file_name.startswith(files_prefix)
                and file_name.endswith(data_ext)
                and patient_name in patient):
            if patient_name not in data_structure:
                data_structure[patient_name] = [file_path]
            else:
                data_structure[patient_name].append(file_path)
    return data_structure


def _get_parsed_data_structure(data_path, files_suffix, files_prefix):
    """"""
    patient = _get_all_patient()
    data_structure = {}
    for patient_name in listdir(data_path):
        patient_path = join(data_path, patient_name)
        if isdir(patient_path) and patient_name in patient:
            data_structure[patient_name] = []
            for patient_file in listdir(patient_path):
                patient_file_path = join(patient_path, patient_file)
                if (isfile(patient_file_path) and
                        patient_file.startswith(files_prefix) and
                        patient_file.endswith(files_suffix)):
                    data_structure[patient_name].append(
                        patient_file_path)
    return data_structure


#   FILESYSTEM: MODELS
def get_model_settings_key():
    """"""
    return _MODEL_SETTINGS_FILE_KEY


def get_model_weight_key():
    """"""
    return _MODEL_WEIGHT_FILE_KEY


def get_model_structure_key():
    """"""
    return _MODEL_STRUCTURE_FILE_KEY


def _get_models_path():
    """"""
    return _DEFAULT_MODELS_PATH


def get_model_path(model_name, models_path=None):
    """"""
    if models_path is None:
        models_path = _get_models_path()
    return join(models_path, model_name)


def get_models_paths(problem, models_path=None):
    """"""
    if models_path is None:
        models_path = _get_models_path()
    return [get_model_path(model_name=model_name,
                           models_path=models_path) for model_name
            in listdir(models_path) if model_name.startswith(
            get_prefix(problem=problem, is_model=True))]


def get_model_out_file_path(model_name, model_path=None,
                            models_path=None):
    """"""
    if model_path is None:
        model_path = get_model_path(model_name, models_path)
    model_file = {}
    for model_file_key, model_file_name in _MODEL_FILE_NAME.items():
        model_file_name = model_name + _MODEL_FILE_EXT[model_file_key]
        model_file[model_file_key] = join(model_path, model_file_name)
    return model_file


def get_new_model_name(problem, models_path=None):
    """"""
    if models_path is None:
        models_path = _get_models_path()
    model_name_prefix = get_prefix(problem=problem, is_model=True)
    return (model_name_prefix
            + str(_get_highest_id(models_path,
                                  file_prefix=model_name_prefix) + 1))


def _get_highest_id(path, file_prefix='', file_suffix=''):
    """"""
    max_id = -1
    if isdir(path):
        for file_name in listdir(path):
            if (file_name.startswith(file_prefix) and
                    file_name.endswith(file_suffix)):
                max_id = max(max_id, int(file_name[len(
                    file_prefix): file_name.rfind(file_suffix)]))
    return max_id


#   DATA INFO
def get_data_info_path(problem, data_path=None):
    """"""
    if data_path is None:
        data_path = _get_data_path(raw_data=False)
    return join(data_path, get_prefix(problem) +
                _DEFAULT_DATA_INFO_FILE)


#   FILESYSTEM: GENERAL
def get_prefix(problem, raw_data=False, is_model=False):
    """"""
    if is_model:
        prefix = (_MODEL_PREFIX + _PATH_WORD_DELIMITER + problem +
                  _PATH_WORD_DELIMITER)
    elif raw_data or problem == 'all':
        prefix = ''
    else:
        prefix = problem + _PATH_WORD_DELIMITER
    return prefix


#   SUMMARY FILE
def get_summary_headers():
    """"""
    return _SUMMARY_HEADERS


def get_summary_file_path(problem, path=None, file_name=None):
    """"""
    if path is None:
        path = _DEFAULT_SUMMARY_PATH
    if file_name is None:
        file_name = _DEFAULT_SUMMARY_FILE_NAME
    return join(path, get_prefix(problem=problem) + file_name)


#   PATIENT FILES
def _get_all_patient():
    """Get list of patient"""
    return _CORRECT_PATIENTS


def get_patient_path(patient_name, data_path=None):
    """"""
    if data_path is None:
        data_path = _get_data_path(raw_data=False)
    return join(data_path, patient_name)


def get_patient_partition(problem):
    """"""
    # TODO: Do something different if the problem is fog or walk
    return [[patient for patient in _CORRECT_PATIENTS if patient
            not in _VAL_PATIENT_DEFAULT
             and patient not in _TEST_PATIENT_DEFAULT],
            _VAL_PATIENT_DEFAULT, _TEST_PATIENT_DEFAULT]


def get_new_patient_file_path(problem, patient_name,
                              raw_file_name,
                              patient_data_path=None):
    """"""
    if patient_data_path is None:
        patient_data_path = get_patient_path(patient_name)
    raw_patient_file_suffix = raw_file_name[raw_file_name.find(
        _PATH_WORD_DELIMITER): raw_file_name.rfind(_get_data_ext(
        raw_data=True))]
    prefix_file_name = (get_prefix(problem=problem) + patient_name
                        + raw_patient_file_suffix
                        + _PATH_WORD_DELIMITER)
    patient_file_ext = _get_data_ext(raw_data=False)
    return join(patient_data_path, prefix_file_name + str(
        _get_highest_id(patient_data_path,
                        file_prefix=prefix_file_name,
                        file_suffix=patient_file_ext) + 1) +
                patient_file_ext)


#   OTHER META-DATA
def get_activity_class():
    """"""
    return _ACTIVITY_CLASS_KEY


#   METRICS KEYS
def get_metric():
    """"""
    return _METRIC_KEYS


def parse_conf_mat(conf_mat):
    """"""
    return OrderedDict(
        [('TP', int(conf_mat[0, 0])), ('FN', int(conf_mat[0, 1])),
         ('FP', int(conf_mat[1, 0])), ('TN', int(conf_mat[1, 1]))])


#   CONFIGURATION PROPERTIES KEYS
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

# EOF

"""Experiment configuration"""

# Authors: Julia Camps <julia.camps.sereix@est.fib.upc.edu>
# Created on: 30/11/2016 09:26

import numpy as np

from FOG.utils import degree_to_radian
from FOG.utils import calc_window_size
from FOG.utils import calc_batch_size


# STATIC PROPERTIES
_N_CHANNEL = 1
_N_FEATURE = 9
_N_CLASS = 2
_DATA_FREQ = 50
_MAX_BATCH_SIZE = 128
_MAX_BATCH_SIZE_TEMPORAL = 64
_SEED = 77
_AUGMENT_SHIFT = 2
_AUGMENT_ROTATE = 4
_ROT_RANGE_ANGLES = [30, 45, 10]
_FILTER_THRESHOLD = 0.4

# EXPERIMENT-CONFIGURATION PROPERTIES
# _INITIALIZATION = ['glorot_uniform', 'lecun_uniform', 'he_normal',
#                   'he_uniform', 'glorot_normal']
_INITIALIZATION = ['glorot_uniform']  # , 'glorot_normal']
_N_CONV_LAYER = [4]
_N_DENSE_LAYER = [1]
_KERNEL_SHAPE = [[32, 11], [32, 5], [64, 3], [64, 3]]
_DENSE_SHAPE = [128]
_DROPOUT = [0.5]  #.25]  # , 0.5]
_OPTIMIZATION = ['adam']
_POOLING = [False]  # , True]
_ATROUS = [False]  # , True]
_REGULARIZATION = [None]  # , 'l1', 'l2']
_TEMPORAL_MODEL = [False]  # False,
_LEARNING_RATE = [0.000001, 0.000005, 0.00001]

# RUN SETTINGS
_LOG_MODE_ID = 0  # {0: 'verbose', 1: 'errors', 2: 'quiet'}
_VERBOSE = True


# DATA-CONFIGURATION PROPERTIES
_DATA_CONFIG = [
    ('Conf_2', {
        'is_temporal': False,
        'window_time': 2,
        'n_train': 188288,
        'n_validation': 6464,
        'n_test': 4032,
        'percent_pos_train': 0.15,
        'percent_pos_val': 0.13,
        'percent_pos_test': 0.25
    }),
    ('Conf_2', {
        'is_temporal': True,
        'window_time': 2,
        'n_train': 115200,
        'n_validation': 3968,
        'n_test': 1920,
        'percent_pos_train': 0.07,
        'percent_pos_val': 0.05,
        'percent_pos_test': 0.07
    })
]


def is_verbose():
    """"""
    return _VERBOSE


def get_log_mode_id():
    """"""
    return _LOG_MODE_ID
     

def get_rotate_range():
    """"""
    return [degree_to_radian(rot_angle)
            for rot_angle in np.array(_ROT_RANGE_ANGLES)]


def experiment_conf_generator():
    """"""
    for temporal in _TEMPORAL_MODEL:
        if temporal:
            max_batch_size = _MAX_BATCH_SIZE_TEMPORAL
        else:
            max_batch_size = _MAX_BATCH_SIZE
        for conf_name, data_conf in [
                (conf_n, d_conf) for conf_n, d_conf in _DATA_CONFIG
                if d_conf['is_temporal'] == temporal]:
            window_time = data_conf['window_time']
            n_train = data_conf['n_train']
            n_validation = data_conf['n_validation']
            n_test = data_conf['n_test']
            percent_pos_train = data_conf['percent_pos_train']
            percent_pos_val = data_conf['percent_pos_val']
            percent_pos_test = data_conf['percent_pos_test']
            window_size = calc_window_size(_DATA_FREQ, window_time)
            batch_size = calc_batch_size(max_batch_size, window_time)
            class_weight = {0: (percent_pos_train * 100),
                            1: ((1-percent_pos_train) * 100)}
            for weight_init in _INITIALIZATION:
                for optimization in _OPTIMIZATION:
                    for learning_rate in _LEARNING_RATE:
                        for n_conv in _N_CONV_LAYER:
                            for n_dense in _N_DENSE_LAYER:
                                for dropout in _DROPOUT:
                                    for pooling in _POOLING:
                                        for regular in _REGULARIZATION:
                                            for atrous in _ATROUS:
                                                yield ({
                                                    'conf_name':
                                                        conf_name,
                                                    'window_time':
                                                        window_time,
                                                    'n_train':
                                                        n_train,
                                                    'n_validation':
                                                        n_validation,
                                                    'n_test': n_test,
                                                    'percent_pos_train':
                                                        percent_pos_train,
                                                    'percent_pos_val':
                                                        percent_pos_val,
                                                    'percent_pos_test':
                                                        percent_pos_test,
                                                    'weight_init':
                                                        weight_init,
                                                    'optimization':
                                                        optimization,
                                                    'n_conv': n_conv,
                                                    'n_dense':
                                                        n_dense,
                                                    'dropout':
                                                        dropout,
                                                    'pooling':
                                                        pooling,
                                                    'atrous': atrous,
                                                    'regular':
                                                        regular,
                                                    'kernel':
                                                        _KERNEL_SHAPE,
                                                    'dense':
                                                        _DENSE_SHAPE,
                                                    'window_size':
                                                        window_size,
                                                    'batch_size':
                                                        batch_size,
                                                    'temporal':
                                                        temporal,
                                                    'class_weight':
                                                        class_weight,
                                                    'learning_rate':
                                                        learning_rate
                                                })


def get_filter_threshold():
    """"""
    return _FILTER_THRESHOLD


def get_n_feature():
    """"""
    return _N_FEATURE


def get_seed_for_random():
    """"""
    return _SEED


def get_shift_augment():
    """"""
    return _AUGMENT_SHIFT


def get_rotate_augment():
    """"""
    return _AUGMENT_ROTATE


# EOF

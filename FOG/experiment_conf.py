"""Experiment configuration"""

# Authors: Julia Camps <julia.camps.sereix@est.fib.upc.edu>
# Created on: 30/11/2016 09:26

import numpy as np

from FOG.utils import degree_to_radian


# STATIC PROPERTIES
_N_CHANNEL = 1
_N_FEATURE = 9
_N_CLASS = 2
_DATA_FREQ = 100
_MAX_BATCH_SIZE = 128
_SEED = 77
_AUGMENT_SHIFT = 2
_AUGMENT_ROTATE = 4
_ROT_RANGE_ANGLES = [30, 45, 10]

# EXPERIMENT-CONFIGURATION PROPERTIES
_INITIALIZATION = ['glorot_uniform', 'lecun_uniform', 'he_normal',
                  'he_uniform', 'glorot_normal']
_N_CONV_LAYER = [4]
_N_DENSE_LAYER = [1]
_KERNEL_SHAPE = [[32, 11], [32, 5], [64, 3], [64, 3]]
_DENSE_SHAPE = [128]
_DROPOUT = [0.25, 0.5]
_OPTIMIZATION = ['adadelta']
_POOLING = [True, False]
_ATROUS = [True, False]
_REGULARIZATION = [None, 'l1', 'l2']

# RUN SETTINGS
_LOG_MODE_ID = 0 # {0: 'verbose', 1: 'errors', 2: 'quiet'}
_VERBOSE = True


# DATA-CONFIGURATION PROPERTIES
_DATA_PROPERTIES = {
    'Conf_2_5': {
        'Conf': {
            'Time': 2,
            'Filter': 0.5
        },
        'Result': {
            'n_train': 185984,
            'n_val': 6400,
            'n_test': 3968,
            'percent_train': 0.14,
            'percent_val': 0.13,
            'percent_test': 0.24
        }
    },
    'Conf_3_5': {
        'Conf': {
            'Time': 3,
            'Filter': 0.5
        },
        'Result': {
            'n_train': 122304,
            'n_val': 4160,
            'n_test': 2560,
            'percent_train': 0.15,
            'percent_val': 0.13,
            'percent_test': 0.24
        }
    }
}


def is_verbose():
    """"""
    return _VERBOSE


def _get_log_mode_id():
    """"""
    return _LOG_MODE_ID
     

def _get_rotate_range():
    """"""
    return [[-degree_to_radian(rot_angle), degree_to_radian(rot_angle)
             ] for rot_angle in np.array(_ROT_RANGE_ANGLES)]


def get_experiment_conf():
    """"""
    experiment_conf_list = []
    for data_conf in _DATA_PROPERTIES:
        for weight_init in _INITIALIZATION:
            for optimization in _OPTIMIZATION:
                for n_conv in _N_CONV_LAYER:
                    for n_dense in _N_DENSE_LAYER:
                        for dropout in _DROPOUT:
                            for pooling in _POOLING:
                                for atrous in _ATROUS:
                                    for regular in _REGULARIZATION:
                                        experiment_conf_list.append({
                                            'data_conf': data_conf,
                                            'weight_init':
                                                weight_init,
                                            'optimization':
                                                optimization,
                                            'n_conv': n_conv,
                                            'n_dense': n_dense,
                                            'dropout': dropout,
                                            'pooling': pooling,
                                            'atrous': atrous,
                                            'regular': regular,
                                            'kernel': _KERNEL_SHAPE,
                                            'dense': _DENSE_SHAPE})
    return experiment_conf_list


def _get_num_channel():
    """"""
    return _N_CHANNEL


def _get_n_feature():
    """"""
    return _N_FEATURE


def _get_n_class():
    """"""
    return _N_CLASS


def _get_random_seed():
    """"""
    return _SEED


def _get_shift_augment():
    """"""
    return _AUGMENT_SHIFT


def _get_rotate_augment():
    """"""
    return _AUGMENT_ROTATE


def _get_freq():
    """"""
    return _DATA_FREQ


def _get_max_batch_size():
    """"""
    return _MAX_BATCH_SIZE


# EOF

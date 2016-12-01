"""Utils functions"""

# Authors: Julia Camps <julia.camps.sereix@est.fib.upc.edu>
# Created on: 29/11/2016 11:26

import numpy as np
import datetime

from FOG.definitions import _get_data_path
from FOG.definitions import _get_delimiter


def is_numeric(string):
    """"""
    if string.startswith('-'):
        res = ((string[1:]).replace('.', '', 1)).isnumeric()
    else:
        res = (string.replace('.', '', 1)).isnumeric()
    return res


def parse_value(content):
    """"""
    if is_numeric(content):
        content = float(content)
    return content


def substract_mean(X):
    """"""
    return X - X.mean(axis=0)


def calc_batch_size(max_size, window_time):
    """"""
    return int(round(max_size / 2 ** (window_time - 1)))


def degree_to_radian(degree):
    """"""
    return degree * np.pi / 180


def plot_results():
    """"""
    
    
def get_date():
    """"""
    return str(datetime.date.today())


def conf_mat_to_str(conf_mat, delimiter=_get_delimiter()):
    """"""
    return ('TP' + delimiter + str(int(conf_mat[0, 0])) + delimiter
            + 'FN' + delimiter + str(int(conf_mat[0, 1])) + delimiter
            + 'FP' + delimiter + str(int(conf_mat[1, 0])) + delimiter
            + 'TN' + delimiter + str(int(conf_mat[1, 1])))

# EOF

"""Utils functions"""

# Authors: Julia Camps <julia.camps.sereix@est.fib.upc.edu>
# Created on: 29/11/2016 11:26

import numpy as np

from FOG.definitions import get_data_path
from FOG.definitions import get_delimiter


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

# EOF

"""Utils functions"""

# Authors: Julia Camps <julia.camps.sereix@est.fib.upc.edu>
# Created on: 29/11/2016 11:26

import numpy as np
import datetime
from collections import OrderedDict

from FOG.definitions import get_delimiter
from FOG.definitions import get_delimiter_level


def is_numeric(string):
    """"""
    if string.endswith('\n'):
        string = string[:-1]
    if string.startswith('-'):
        res = ((string[1:]).replace('.', '', 1)).isnumeric()
    else:
        res = (string.replace('.', '', 1)).isnumeric()
    return res


def parse_value(content):
    """"""
    if content.endswith('\n'):
        content = content[:-1]
    if is_numeric(content):
        content = float(content)
    return content


def substract_mean(X):
    """"""
    return X - X.mean(axis=0)


def calc_batch_size(max_size, window_time):
    """"""
    return int(round(max_size / 2 ** (window_time - 1)))


def calc_window_size(freq, window_time):
    """"""
    return freq * window_time


def degree_to_radian(degree):
    """"""
    return degree * np.pi / 180


def plot_results():
    """"""
    # TODO
    
    
def get_date():
    """"""
    return str(datetime.date.today())


def split_data(X):
    """"""
    return (np.asarray(X[:, :-2]), np.asarray(X[:, -2]),
            np.asarray(X[:, -1]))


def to_string(data, delimiter_level=None, inter_delimiter=None,
              intra_delimiter=None, it_delimiter=0):
    """"""
    if (delimiter_level is None or inter_delimiter is None or
            intra_delimiter is None):
        [delimiter_level, inter_delimiter, intra_delimiter] \
            = get_delimiter()
    if isinstance(data, tuple):
        data_str = (delimiter_level[it_delimiter].join(
            [to_string(data_part, delimiter_level=delimiter_level,
                       inter_delimiter=inter_delimiter,
                       intra_delimiter=intra_delimiter,
                       it_delimiter=it_delimiter) for data_part in data]))
    elif isinstance(data, (dict, OrderedDict)):
        data_str = (delimiter_level[it_delimiter].join(
            [inter_delimiter.join([to_string(key_part,
                                        delimiter_level=delimiter_level,
                       inter_delimiter=inter_delimiter,
                       intra_delimiter=intra_delimiter,
                       it_delimiter=1), to_string(data_part,
                                 delimiter_level=delimiter_level,
                       inter_delimiter=inter_delimiter,
                       intra_delimiter=intra_delimiter,
                       it_delimiter=1)])
             for key_part, data_part in data.items()]))
    elif isinstance(data, (list, tuple, np.ndarray)):
        data_str = (delimiter_level[it_delimiter].join(
            [to_string(data_part, delimiter_level=delimiter_level,
                       inter_delimiter=inter_delimiter,
                       intra_delimiter=intra_delimiter,
                       it_delimiter=1) for data_part in data]))
    else:
        data_str = (str(data).replace(inter_delimiter,
                                       intra_delimiter))
    return data_str


def from_string(data_str, inter_delimiter=None, intra_delimiter=None):
    """"""
    if inter_delimiter is None or intra_delimiter is None:
        inter_delimiter, intra_delimiter = get_delimiter_level(
            n_level=2)
    if data_str.find(inter_delimiter) > -1:
        data = OrderedDict([from_string(data_str_part,
                            inter_delimiter=inter_delimiter,
                            intra_delimiter=intra_delimiter)
                            for data_str_part in data_str.split(
                inter_delimiter)])
        
    elif data_str.find(intra_delimiter) > -1:
        
        data = (from_string(data_str[:data_str.find(
            intra_delimiter)], inter_delimiter=inter_delimiter,
                            intra_delimiter=intra_delimiter),
                [from_string(data_str_part,
                            inter_delimiter=inter_delimiter,
                            intra_delimiter=intra_delimiter)
                for data_str_part in data_str[data_str.find(
            intra_delimiter):].split(intra_delimiter)]
                )
    else:
        data = parse_value(data_str)
    return data



# EOF

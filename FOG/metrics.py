"""Metrics and statistics calculating functions"""

# Authors: Julia Camps <julia.camps.sereix@est.fib.upc.edu>
# Created on: 18/11/2016 20:55

import numpy as np

from collections import Counter
from collections import OrderedDict

from FOG.definitions import parse_conf_mat
from FOG.definitions import get_metric
from FOG.definitions import get_activity_class
from FOG.definitions import label_is_positive
from FOG.definitions import label_is_negative


_METRICS = get_metric()


# def metrics_calc(y_true, y_pred, metrics=None):
#     """"""
#     if metrics is None:
#         metrics = _METRICS
#     result_metrics = OrderedDict([])
#     for metric in metrics:
#         result_metrics[metric] = _calc_metric(conf_mat, metric)
#     return result_metrics


def _calc_metric(conf_mat, metric_name):
    """"""
    metric_value = None
    if metric_name.find('conf') > -1:
        metric_value = conf_mat
    else:
        TP = conf_mat['TP']
        FN = conf_mat['FN']
        FP = conf_mat['FP']
        TN = conf_mat['TN']
        if metric_name.find('acc') > -1:
            if TP > 0 or TN > 0:
                metric_value = (TP + TN) / (TP + FN + FP + TN)
        elif metric_name.find('sen') > -1:
            if TP > 0:
                metric_value = TP / (TP + FN)
        elif metric_name.find('spe') > -1:
            if TN > 0:
                metric_value = TN / (TN + FP)
    return metric_value


def get_statistics(label_data):
    """"""
    existing_class = get_activity_class()
    raw_conf_mat_total = np.zeros((2, 2))
    raw_class_statistic = OrderedDict(
        [(class_key, np.zeros((2, 2))) for class_key in
         existing_class])
    
    for [y_true, y_orig, y_pred] in label_data:
        [conf_mat, raw_class_statistic_part] = statistics_matrixes(
            y_true, y_pred, y_orig)
        raw_conf_mat_total += conf_mat
        for key, raw_class_conf_mat in \
                raw_class_statistic_part.items():
            raw_class_statistic[key] += raw_class_conf_mat
        
    return [parse_conf_mat(raw_conf_mat_total),
            OrderedDict([(class_key, parse_conf_mat(
                raw_class_conf_mat))
                         for class_key, raw_class_conf_mat in
                         raw_class_statistic.items()])]


def statistics_matrixes(y_true, y_pred, y_orig):
    """"""
    y_orig_counter = []
    for y_orig_part in y_orig:
        y_orig_counter.append(Counter(y_orig_part))
        
    raw_conf_mat = np.zeros((2, 2))
    class_statistic = {}
    for i in range(np.asarray(y_true).shape[0]):
        y_orig_counter_part = y_orig_counter[i]
        if label_is_positive(int(y_true[i])):
            if int(y_true[i]) == int(np.round(y_pred[i])):
                raw_conf_mat[0, 0] += 1
                for key, count in y_orig_counter_part.items():
                    if key not in class_statistic:
                        class_statistic[key] = np.zeros((2, 2))
                    class_statistic[key][0, 0] += count
            else:
                raw_conf_mat[0, 1] += 1
                for key, count in y_orig_counter_part.items():
                    if key not in class_statistic:
                        class_statistic[key] = np.zeros((2, 2))
                    class_statistic[key][0, 1] += count
        elif label_is_negative(int(y_true[i])):
            if int(y_true[i]) == int(np.round(y_pred[i])):
                raw_conf_mat[1, 1] += 1
                for key, count in y_orig_counter_part.items():
                    if key not in class_statistic:
                        class_statistic[key] = np.zeros((2, 2))
                    class_statistic[key][1, 1] += count
            else:
                raw_conf_mat[1, 0] += 1
                for key, count in y_orig_counter_part.items():
                    if key not in class_statistic:
                        class_statistic[key] = np.zeros((2, 2))
                    class_statistic[key][1, 0] += count
    return [raw_conf_mat, class_statistic]


# EOF

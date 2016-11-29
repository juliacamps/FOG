"""Metrics and statistics calculating functions"""

# Authors: Julia Camps <julia.camps.sereix@est.fib.upc.edu>
# Created on: 18/11/2016 20:55

import numpy as np

from FOG.preprocessing_tools import check_orig
from FOG.definitions import get_activity_class
from FOG.definitions import get_existing_metric
from FOG.definitions import get_metric_init


def metrics_calc(conf_mat, metrics=get_existing_metric()):
    """"""
    TP = conf_mat[0, 0]
    FN = conf_mat[0, 1]
    FP = conf_mat[1, 0]
    TN = conf_mat[1, 1]
    result_metrics = {}
    if 'accuracy' in metrics:
        if TP > 0 or TN > 0:
            result_metrics['accuracy'] = (TP + TN) / (TP + FN + FP
                                                      + TN)
        else:
            result_metrics['accuracy'] = 0
    if 'sensitivity' in metrics:
        if TP > 0:
            result_metrics['sensitivity'] = TP / (TP + FN)
        else:
            result_metrics['sensitivity'] = 0
    if 'specificity' in metrics:
        if TN > 0:
            result_metrics['specificity'] = TN / (TN + FP)
        else:
            result_metrics['specificity'] = 0
    return result_metrics


def get_statistics(model_, generator, samples_count, batch_size, msg):
    """"""
    existing_class = get_activity_class()
    existing_metric = get_existing_metric()
    confusion_mat = np.zeros((2, 2))
    status = 'OK'
    end_epoch = False
    samples_it = 0
    partial_statistic = {}
    for activity_class in existing_class:
        partial_statistic[activity_class] = {}
        for metric in existing_metric:
            partial_statistic[activity_class][metric] = \
                get_metric_init(metric)
    
    for X, y_true, y_origs in generator:
        try:
            y_pred = model_.predict_on_batch(X)
        except Exception as e:
            status = ('ERROR: While ' + msg + ': ' + str(repr(e)))
            end_epoch = True
        else:
            [conf_mat, class_statistic] = statistics_matrixes(
                y_true, y_pred, y_origs)
            confusion_mat += conf_mat
            for class_key, class_conf_mat in class_statistic.items():
                partial_statistic[class_key]['conf_mat'] \
                    += class_conf_mat
            samples_it += batch_size
        
        if end_epoch or samples_it >= samples_count:
            # End loop condition (only condition)
            break
    
    for activity_class in existing_class:
        conf_mat = partial_statistic[activity_class]['conf_mat']
        metrics = metrics_calc(conf_mat, metrics=existing_metric)
        for metric_name, metric_value in metrics.items():
            partial_statistic[activity_class][metric_name] \
                = metric_value
    return [status, confusion_mat, partial_statistic]


def statistics_matrixes(y_true, y_pred, y_origs):
    """"""
    y_origs_counter = []
    for y_orig in y_origs:
        y_origs_counter.append(check_orig(y_orig))
        
    conf_mat = np.zeros((2, 2))
    statistic_mat = {}
    for i in range(np.asarray(y_true).shape[0]):
        y_orig_counter = y_origs_counter[i]
        if int(y_true[i]) == 1:
            if int(y_true[i]) == int(np.round(y_pred[i])):
                conf_mat[0, 0] += 1
                for k, count in y_orig_counter.items():
                    if k not in statistic_mat:
                        statistic_mat[k] = np.zeros((2, 2))
                    statistic_mat[k][0, 0] += count
            else:
                conf_mat[0, 1] += 1
                for k, count in y_orig_counter.items():
                    if k not in statistic_mat:
                        statistic_mat[k] = np.zeros((2, 2))
                    statistic_mat[k][0, 1] += count
        else:
            if int(y_true[i]) == int(np.round(y_pred[i])):
                conf_mat[1, 1] += 1
                for key, count in y_orig_counter.items():
                    if key not in statistic_mat:
                        statistic_mat[key] = np.zeros((2, 2))
                    statistic_mat[key][1, 1] += count
            else:
                conf_mat[1, 0] += 1
                for k, count in y_orig_counter.items():
                    if k not in statistic_mat:
                        statistic_mat[k] = np.zeros((2, 2))
                    statistic_mat[k][1, 0] += count
    return [conf_mat, statistic_mat]

# EOF
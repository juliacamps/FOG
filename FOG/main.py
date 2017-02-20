"""CNN for walking detection"""

# Authors: Julia Camps <julia.camps.sereix@est.fib.upc.edu>
# Created on: 06/10/2016 17:22

import numpy as np
import random as rd
from collections import OrderedDict, defaultdict

from FOG.definitions import init_settings
from FOG.definitions import define_settings
from FOG.definitions import get_new_model_name
from FOG.definitions import get_model_names
from FOG.definitions import check_status

from FOG.utils import get_date
from FOG.utils import to_string

from FOG.io_functions import save_model
from FOG.io_functions import report_event
from FOG.io_functions import load_model
from FOG.io_functions import save_prediction

from FOG.experiment_conf import get_seed_for_random
from FOG.experiment_conf import experiment_conf_generator

from FOG.preprocessing_tools import get_patient_split
from FOG.preprocessing_tools import get_dataset
from FOG.preprocessing_tools import get_data_files

from FOG.models import build_model
from FOG.models import compile_model

from FOG.core_functions import train_model
from FOG.core_functions import predict_model
from FOG.core_functions import predict_result
from FOG.core_functions import add_configuration


_DETECTION_PROBLEM = 'fog'
_GENERATE_SUMMARY = False
_TRAIN_MODEL = True
_TEST_MODEL = False
_REPRODUCIBILITY = False
_PREDICT = True
_SAVE_REDUCED_PREDICTION = True

_MODEL_NAME = [
    'model_fog_55',
    'model_fog_56',
    'model_fog_57',
    'model_fog_58'
    # 'model_fog_59'
]
_SEED = get_seed_for_random()

def predict(model_name, train_patient, val_patient, test_patient):
    """"""
    status, model, settings, configuration = load_model(model_name)
    
    compile_model(model, configuration['objective'],
                  configuration['optimization'])
    
    add_configuration(model_name, configuration)
    
    # Load data
    data = OrderedDict()
    data['train'] = get_data_files(train_patient)
    data['val'] = get_data_files(val_patient)
    data['test'] = get_data_files(test_patient)
    
    predict_model(model, data,
                  batch_size=configuration['batch_size'],
                  window_size=configuration['window_size'],
                  temporal=configuration['temporal'],
                  percent_throw_no_fog=configuration[
                      'percent_throw_no_fog'], cutting=configuration[
            'cutting'],
                  pure_threshold=configuration[
                      'pure_threshold'],
                  problem=_DETECTION_PROBLEM,
                  model_name=model_name,
                  reduce_memory=_SAVE_REDUCED_PREDICTION)

if __name__ == '__main__':
    
    # Initialize
    if _REPRODUCIBILITY:
        np.random.seed(_SEED)
        rd.seed(_SEED)

    # Get data
    [train_patient, val_patient, test_patient] = get_patient_split(
        _DETECTION_PROBLEM)
    
    # PREDICT AND PLOT RESULTS
    if _PREDICT and not _TRAIN_MODEL:
        for model_name in _MODEL_NAME:
            
            settings = init_settings()
            status, model, settings_aux, configuration = load_model(
                model_name)
            
            settings = define_settings(settings,
                                       new_settings_dict=configuration)
            compile_model(model, configuration['objective'],
                          configuration['optimization'])

            add_configuration(model_name, configuration)
            

            # Load data
            data = OrderedDict()
            data['train'] = get_data_files(train_patient)
            data['val'] = get_data_files(val_patient)
            data['test'] = get_data_files(test_patient)

            predict_model(model, data,
                          batch_size=configuration['batch_size'],
                          window_size=configuration['window_size'],
                          temporal=configuration['temporal'],
                          percent_throw_no_fog=configuration[
                              'percent_throw_no_fog'], cutting=configuration[
                              'cutting'],
                          pure_threshold=configuration[
                              'pure_threshold'],
                          problem=_DETECTION_PROBLEM,
                          model_name=model_name,
                          reduce_memory=_SAVE_REDUCED_PREDICTION)
        
    # NEW MODEL CODE
    if _TRAIN_MODEL:
        # Load data
        train_data = get_dataset(train_patient)
        validation_data = get_dataset(val_patient)
        
        exp_index = 0
        load_m = False  # _LOAD_MODEL[exp_index]
        for configuration in experiment_conf_generator():
            
            day_date = get_date()
            settings = init_settings()
            settings = define_settings(settings, date=day_date,
                                       new_settings_dict=configuration)
            n_epoch = configuration['n_epoch']

            
            if not load_m:
                model_name = get_new_model_name(
                    problem=_DETECTION_PROBLEM)
                settings = define_settings(
                    settings, model_name=model_name,
                    problem=_DETECTION_PROBLEM, n_epoch=n_epoch,
                    reproducibility=_REPRODUCIBILITY,
                    random_seed=_SEED)
            
                [model_structure, model] = build_model(
                    window_size=configuration['window_size'],
                    conv_layers=configuration['conv'],
                    dense_layers=configuration['dense'],
                    optimization=configuration['optimization'],
                    pooling_layers=configuration['pooling'],
                    dropout=configuration['dropout'],
                    init=configuration['weight_init'],
                    atrous=configuration['atrous'],
                    regularizer_conf=configuration['regularizer'],
                    temporal=configuration['temporal'],
                    objective=configuration['objective'],
                    activation=configuration['activation'],
                    last_layer=configuration['last_layer'])
                
                settings = define_settings(
                    settings, model_conf=model_structure,
                    n_parameter=model.count_params())
            else:
                model_name = _MODEL_NAME[exp_index]
                status, model, settings_aux = load_model(model_name)
                compile_model(model, configuration['objective'],
                              configuration['optimization'])
                
            exp_index += 1

            msg = ('\n\n============= NEW EXPERIMENT ============='
                   + '\nConfiguration:\n' + to_string(settings))
            report_event(msg, is_run_log=True)

            model, train_summary, settings, n_epoch_real = \
                train_model(
                model, train_data, n_epoch, configuration[
                    'n_train'], configuration['batch_size'],
                configuration['window_size'],
                configuration['temporal'], _DETECTION_PROBLEM,
                configuration['percent_throw_no_fog'], configuration['shift'],
                configuration['rotate'], configuration[
                    'pos_threshold'], configuration['cutting'],
                configuration['pure_threshold'],
                validation_data=validation_data,
                n_validation=configuration['n_validation'],
                settings=settings, log_file_name=(
                    'train_log/' + model_name +
                        configuration['conf_name'] + '.csv'),
                model_name=model_name)
            configuration['n_epoch'] = n_epoch_real
            save_model(model, model_name, settings, configuration)

            msg = ('============ RESULTS & SETTINGS ============'
                   + '\nConfiguration:\n' + to_string(settings)
                   + '\n============ END OF EXPERIMENT ============')
            report_event(msg, is_run_log=True)
            
            # IF TRAIN AND PREDICT ON NEW MODEL
            if _PREDICT:
                predict(model_name, train_patient, val_patient,
                        test_patient)
                       
    # SUMMARY GENERATION CODE
    if _GENERATE_SUMMARY:
        prediction = OrderedDict([])
        for model_name in get_model_names(_DETECTION_PROBLEM):
            warning_msg_header = ('WARNING: ' + model_name
                           + ' has been discarded from the '
                           + 'evaluation process, due to -> ')
            msg = ('OK: Prediction for model ' + model_name
                   + ' successful')
            is_error = False
            status, model, settings = load_model(model_name)
            if check_status(status):
                status = settings['final_status']
                if check_status(status):
                    if settings['reproducibility']:
                        np.random.seed(settings['random_seed'])
                        rd.seed(settings['random_seed'])
                        report_event(
                            'STARTING EVALUATION FOR MODEL ' +
                            model_name, is_run_log=True)
                        status, prediction, summary = predict_result(
                            model, settings['train_patient'],
                            settings['window_size'],
                            settings['batch_size'],
                            settings['temporal'], settings['problem'],
                            settings['n_train'],
                            settings['threshold'],
                            settings['augment_shift'],
                            settings['augment_rotate'],
                            validation_patient=settings[
                                'validation_patient'],
                            n_validation=settings['n_validation'])
                        if check_status(status):
                            status = save_prediction(settings,
                                                     summary,
                                            settings['problem'])
                            if check_status(status):
                                msg = ('OK: Prediction for model '
                                         + model_name + ' successful')
                                is_error = False
                            else:
                                msg = ('ERROR: Saving prediction for '
                                       + 'model ' + model_name
                                       + ' FAILED!: ' + status)
                                is_error = True
                        else:
                            msg = ('ERROR: Prediction for model '
                                   + model_name + ' FAILED!' + status)
                            is_error = True
                    else:
                        report_event(warning_msg_header
                                     + 'Results can NOT  be '
                                     + 'reproduced!')
                else:
                    msg = (warning_msg_header
                           + ' Loaded status is INVALID!')
                    is_error = True
            else:
                msg = (warning_msg_header
                       + ' Loading process FAILED!: ' + status)
                is_error = True
            report_event(msg, is_error=is_error)

    print('END')

# EOF

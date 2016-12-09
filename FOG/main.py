"""CNN for walking detection"""

# Authors: Julia Camps <julia.camps.sereix@est.fib.upc.edu>
# Created on: 06/10/2016 17:22

import numpy as np
import random as rd
from collections import OrderedDict

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

from FOG.models import build_model

from FOG.core_functions import train_model
from FOG.core_functions import predict_result


_DETECTION_PROBLEM = 'fog'
_GENERATE_SUMMARY = True
_TRAIN_MODEL = True
_TEST_MODEL = False
_REPRODUCIBILITY = True
_N_EPOCH = 200  # x2
_TEMPORAL_STRATEGY = False
_SEED = get_seed_for_random()


if __name__ == '__main__':
    
    # Initialize
    if _REPRODUCIBILITY:
        np.random.seed(_SEED)
        rd.seed(_SEED)

    # Get data
    [train_patient, val_patient, test_patient] = get_patient_split(
        _DETECTION_PROBLEM)

    # Build model
    if _TRAIN_MODEL:
        for configuration in experiment_conf_generator():
            settings = init_settings()
            model_name = get_new_model_name(
                problem=_DETECTION_PROBLEM)
            day_date = get_date()
            
            settings = define_settings(
                settings, new_settings_dict=configuration,
                date=day_date,
                model_name=model_name, problem=_DETECTION_PROBLEM,
                n_epoch=_N_EPOCH, reproducibility=_REPRODUCIBILITY,
                random_seed=_SEED, temporal=_TEMPORAL_STRATEGY)
            
            
            [model_structure, model] = build_model(
                window_size=configuration['window_size'],
                n_conv=configuration['n_conv'],
                n_dense=configuration['n_dense'],
                k_shapes=configuration['kernel'],
                dense_shape=configuration['dense'],
                opt_name=configuration['optimization'],
                pooling=configuration['pooling'],
                dropout=configuration['dropout'],
                init=configuration['weight_init'],
                atrous=configuration['atrous'],
                regularizer=configuration['regular'],
                temporal_model=_TEMPORAL_STRATEGY)
            
            settings = define_settings(
                settings, model_conf=model_structure,
                n_parameter=model.count_params())
            msg = ('\n\n============= NEW EXPERIMENT ============='
                   + '\nConfiguration:\n' + to_string(settings))
            report_event(msg, is_run_log=True)
            
            model, train_summary, settings = train_model(
                model, train_patient, _N_EPOCH, configuration[
                    'n_train'], configuration['batch_size'],
                configuration['window_size'], _TEMPORAL_STRATEGY,
                _DETECTION_PROBLEM, validation_patient=val_patient,
                n_validation=configuration['n_validation'],
                settings=settings)
            
            save_model(model, model_name, (settings, train_summary))
                            
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

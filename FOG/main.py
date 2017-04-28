"""CNN for walking detection"""

# Authors: Julia Camps <julia.camps.sereix@est.fib.upc.edu>
# Created on: 06/10/2016 17:22
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import random as rd
from collections import OrderedDict

from FOG.definitions import init_settings
from FOG.definitions import define_settings
from FOG.definitions import get_new_model_name
from FOG.definitions import get_train_log_path

from FOG.utils import get_date
from FOG.utils import to_string

from FOG.io_functions import save_my_model
from FOG.io_functions import report_event
from FOG.io_functions import load_my_model

from FOG.experiment_conf import get_seed_for_random
from FOG.experiment_conf import experiment_conf_generator

from FOG.preprocessing_tools import get_patient_split
from FOG.preprocessing_tools import get_dataset
from FOG.preprocessing_tools import get_data_files

from FOG.models import build_model
from FOG.models import compile_model

from FOG.core_functions import train_model
from FOG.core_functions import predict_model
from FOG.core_functions import add_configuration
# from FOG.core_functions import generate_all_data
# from FOG.core_functions import load_precalculated
from FOG.core_functions import conf_to_string

_GENERATE_SUMMARY = False
_TRAIN_MODEL = True
_TEST_MODEL = False
_REPRODUCIBILITY = False
_PREDICT = True
_SAVE_REDUCED_PREDICTION = True
_RECALCULATE_DATA = False
_USE_PRECALCULATED = False
_SAVE_TRAIN_ERROR = True
_ONLY_SAVE_SUCCESSFUL = True

_MODEL_NAME = ['model_fog_505']
_SEED = get_seed_for_random()


def predict(model_name):
    """"""
    status, model, settings, configuration = load_my_model(model_name)

    compile_model(model, configuration['objective'], configuration['penalty'], configuration['learning_rate'], configuration['optimizer'])

    add_configuration(model_name, configuration)

    data_freq = configuration['data_freq']

    # Get data
    [train_patient, val_patient,
     test_patient] = get_patient_split(data_freq)
    
    # Load data
    data = OrderedDict()
    data['train'] = get_data_files(train_patient)
    data['val'] = get_data_files(val_patient)
    data['test'] = get_data_files(test_patient)
    
    predict_model(model, data,
                  batch_size=configuration['batch_size'],
                  window_size=configuration['window_size'],
                  temporal=configuration['temporal'],
                  stacking=configuration['stacking'],
                  pure_threshold=configuration[
                      'pure_threshold'],
                  data_freq=configuration['data_freq'],
                  n_feature=configuration['n_feature'],
                  model_name=model_name,
                  reduce_memory=_SAVE_REDUCED_PREDICTION)

if __name__ == '__main__':
    print('miau')
    # Initialize
    if _REPRODUCIBILITY:
        np.random.seed(_SEED)
        rd.seed(_SEED)
    
    
    # PREDICT AND PLOT RESULTS
    if _PREDICT and not _TRAIN_MODEL:
        for model_name in _MODEL_NAME:
            predict(model_name)
        
    # NEW MODEL CODE
    elif _TRAIN_MODEL:
        
        exp_index = 0
        data_freq_ant = None
        for configuration in experiment_conf_generator():
            day_date = get_date()
            settings = init_settings()
            settings = define_settings(settings, date=day_date,
                                       new_settings_dict=configuration)
            n_epoch = configuration['n_epoch']
            data_freq = configuration['data_freq']
            
            if data_freq_ant is None or data_freq != data_freq_ant:
                data_freq_ant = data_freq
            

                [train_patient, val_patient, test_patient] = \
                    get_patient_split(data_freq)

                # Load data
                train_data = get_dataset(train_patient)
                validation_data = get_dataset(val_patient)

            model_name = get_new_model_name()
            settings = define_settings(
                settings, model_name=model_name, n_epoch=n_epoch,
                reproducibility=_REPRODUCIBILITY,
                random_seed=_SEED)
            window_size = configuration['window_size']
            n_feature = configuration[
                    'n_features_per_sample']
            stacking = configuration['stacking']
            spectral_input_size = int(window_size/2)
            temporal_input_size = window_size
            spectral_n_feature = int(n_feature * 2)
            temporal_n_feature = n_feature
            [model_structure, model] = build_model(
                spectral_input_size=spectral_input_size,
                temporal_input_size=temporal_input_size,
                spectral_n_feature=spectral_n_feature,
                temporal_n_feature=temporal_n_feature,
                conv_layers=configuration['conv'],
                dense_layers=configuration['dense'],
                learning_rate=configuration['learning_rate'],
                optimizer=configuration['optimizer'],
                pooling=configuration['pooling'],
                dropout=configuration['dropout'],
                init=configuration['weight_init'],
                atrous=configuration['atrous'],
                regularizer_conf=configuration['regularizer'],
                temporal=configuration['temporal'],
                objective=configuration['objective'],
                penalty=configuration['penalty'],
                activation=configuration['activation'],
                last_layer=configuration['last_layer'],
                batch_size=configuration['batch_size'],
                n_batch_per_file=configuration[
                    'n_batch_per_file'],
                lstm_dropout=configuration['lstm_dropout'])
            
            settings = define_settings(
                settings, model_conf=model_structure,
                n_parameter=model.count_params())

            msg = ('\n\n============= NEW EXPERIMENT ============='
                   + '\nConfiguration:\n' + to_string(settings))
            report_event(msg, is_run_log=True)

            conf_str = conf_to_string(model_name, configuration)

            model, n_epoch_real, success = train_model(
                model, train_data, n_epoch, configuration[
                    'n_train'], configuration['batch_size'],
                configuration['window_size'],
                configuration['stacking'],
                configuration['pure_threshold'],
                    configuration['data_freq'],
                    configuration['augmentation'],
                    configuration['n_batch_per_file'],
                 configuration['n_feature'],
                    configuration['roate_proba'],
                validation_data=validation_data,
                n_validation=configuration['n_validation'],
                settings=settings, log_file_name=(
                        get_train_log_path() + model_name + '_train_log' +
                    '.csv'),
                model_name=model_name, temporal=configuration[
                        'temporal'],
                    save_train_error=_SAVE_TRAIN_ERROR, conf_str=conf_str)
            if success:
                configuration['n_epoch'] = n_epoch_real
                save_my_model(model, model_name, settings,
                            configuration)

                msg = ('============ RESULTS & SETTINGS ============'
                       + '\nConfiguration:\n' + to_string(settings)
                       + '\n============ END OF EXPERIMENT ============')
                report_event(msg, is_run_log=True)
            else:
                msg = ('============ RESULTS & SETTINGS ============'
                       + '\nConfiguration:\n' + to_string(settings)
                       + '\n============ END OF EXPERIMENT ============')
                report_event(msg, is_run_log=True)
            
            # IF TRAIN AND PREDICT ON NEW MODEL
            if _PREDICT and ((not _ONLY_SAVE_SUCCESSFUL) or success):
                predict(model_name)

    print('END')

# EOF

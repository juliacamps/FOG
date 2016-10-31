"""Clean data for FOG and WALKING detection"""

# Authors: Julia Camps <julia.camps.sereix@est.fib.upc.edu>
# Created on: 06/10/2016 17:22

import numpy as np

from src.libraries.io_functions import silent_remove
from src.libraries.io_functions import get_data_path
from src.libraries.io_functions import get_raw_data_path
from src.libraries.io_functions import save_matrix_data
from src.libraries.io_functions import read_data
from src.libraries.io_functions import prepare_path
from src.libraries.preprocessing_tools import full_preprocessing
from src.libraries.preprocessing_tools import generate_dataset

# Environment params and settings
raw_file = get_raw_data_path(group=True)
old_file = get_data_path()

for [old_file_path, old_name] in old_file:
    silent_remove(old_file_path)
    
new_file_name_W = 'walk_'
new_file_name_F = 'fog_'

# Loop for all files to be parsed
n_patient = len(raw_file.items())
info_W = np.zeros([n_patient, 3])
info_F = np.zeros([n_patient, 3])
mean_ = np.zeros([2, 9])
std_ = np.zeros([2, 9])

it_std_n = 0
n_sample_cum = 0
mean_cum = np.zeros(9)
std_cum = np.zeros(9)

patient_names = ''
i = 0
for patient_name, patient_data in raw_file.items():
    new_path = prepare_path(patient_name)
    patient_names += patient_name + '#'
    print('Currently processing data from patient: ' + patient_name)
    for file_path, file_name in patient_data:
        # Read raw data
        mat_contents = read_data(file_path)[0]
        # Reformat data
        giro_data = np.array((mat_contents["sWaist"][0][0])[1][:])
        acc_data = np.array((mat_contents["sWaist"][0][0])[2][:])
        magn_data = np.array((mat_contents["sWaist"][0][0])[3][:])
        class_data_F = np.array((((mat_contents["sWaist"][0][0])[11])
                                 [0][0][10])[:])
        class_data_W = np.array((((mat_contents["sWaist"][0][0])[11])
                                 [0][0][9])[:])
        
        # CLEAN NAN
        sub_part_indexes_raw = np.unique(np.concatenate((np.where(
            np.isnan(giro_data))[0], np.where(np.isnan(acc_data))[0],
            np.where(np.isnan(magn_data))[0],
            np.where(np.isnan(class_data_W))[0],
            np.where(np.isnan(class_data_F))[0])))
        
        sub_part_indexes = []
        if len(sub_part_indexes_raw) > 0:
            pre_index = sub_part_indexes_raw[0]
            sub_part_indexes.append(pre_index)
            if len(sub_part_indexes_raw) > 1:
                next_index = 0
                for next_index in sub_part_indexes_raw:
                    if next_index > (pre_index + 1):
                        sub_part_indexes.append((pre_index + 1))
                        sub_part_indexes.append(next_index)
                    pre_index = next_index
                if next_index < (giro_data.shape[0] - 1):
                    sub_part_indexes.append(next_index + 1)
            else:
                sub_part_indexes.append(pre_index + 1)

        giro_data_part = np.split(giro_data, sub_part_indexes)
        acc_data_part = np.split(acc_data, sub_part_indexes)
        magn_data_part = np.split(magn_data, sub_part_indexes)
        class_data_W_part = np.split(class_data_W, sub_part_indexes)
        class_data_F_part = np.split(class_data_F, sub_part_indexes)
        
        for it_sub_part in range(0, len(giro_data_part), 2):
            sub_giro = giro_data_part[it_sub_part]
            sub_acc = acc_data_part[it_sub_part]
            sub_magn = magn_data_part[it_sub_part]
            sub_W = class_data_W_part[it_sub_part]
            sub_F = class_data_F_part[it_sub_part]
            
            n_sample = sub_W.shape[0]
            n_unlabeled = np.sum((sub_W == 0))
            info_W[i, 0] += n_sample
            info_W[i, 1] += np.sum((sub_W == 6))
            info_W[i, 2] += n_unlabeled
            info_F[i, 0] += n_sample
            info_F[i, 1] += np.sum((sub_F > 1))
            info_F[i, 2] += n_unlabeled

            # CLEANING CRITERIA
            criteria_W = [0, 29, 30, 31]
            indexes_W = np.where(sub_W == criteria_W)[0]
            # No_FoG:   FoG <= 1 & Activity_Posture > 0
            # FoG:      FoG >= 2 & Activity_Posture > 0
            criteria_F = [0]
            indexes_F = np.where(sub_W == criteria_F)[0]
            

            sub_W = (sub_W == 6)
            sub_W[indexes_W] = -1
            data_W = np.concatenate((sub_giro, sub_acc, sub_magn,
                                     sub_W), axis=1)
            sub_F = (sub_F > 1)
            sub_F[indexes_F] = -1
            data_F = np.concatenate((sub_giro, sub_acc, sub_magn,
                                     sub_F), axis=1)
            
            # SAVE DATA
            save_matrix_data(data_W, file_name=(new_file_name_W
                                                + file_name
                                                + str(it_sub_part)),
                             data_path=new_path)
            save_matrix_data(data_F, file_name=(new_file_name_F
                                                + file_name
                                                + str(it_sub_part)),
                             data_path=new_path)
            for std_it in range(n_sample):
                it_std_n += 1
                curr_sample = np.array([sub_giro[std_it], sub_acc[
                    std_it], sub_magn[std_it]]).reshape(9)
                mean_diff = curr_sample - mean_cum
                mean_cum += (mean_diff / it_std_n)
                std_cum += (mean_diff * (curr_sample - mean_cum))
    i += 1
std_cum = np.sqrt(std_cum/(it_std_n - 1))

std_mean_result = [['Data_column', 'giro_1', 'giro_2', 'giro_3',
                    'acc_1', 'acc_2', 'acc_3', 'magn_1', 'magn_2',
                    'magn_3'], ['Population_std', str(std_cum[0]),
                                str(std_cum[1]), str(std_cum[2]),
                                str(std_cum[3]), str(std_cum[4]),
                                str(std_cum[5]), str(std_cum[6]),
                                str(std_cum[7]), str(std_cum[8])],
                   ['Mean', str(mean_cum[0]), str(mean_cum[1]),
                    str(mean_cum[2]), str(mean_cum[3]),
                    str(mean_cum[4]), str(mean_cum[5]),
                    str(mean_cum[6]), str(mean_cum[7]),
                    str(mean_cum[8])]]

# SAVE META-DATA
info_W_result = [std_mean_result[0], std_mean_result[1],
                 std_mean_result[2], ['Patient', 'Instances',
                                      'Positive_count',
                                      'Unlabeled_count',
                                      'Positive_fraction',
                                      'Unlabeled_fraction']]
info_F_result = [std_mean_result[0], std_mean_result[1],
                 std_mean_result[2], ['Patient', 'Instances',
                                      'Positive_count',
                                      'Unlabeled_count',
                                      'Positive_fraction',
                                      'Unlabeled_fraction']]
patient_names = patient_names.split('#')[:-1]
for i in range(len(patient_names)):
    info_W_result.append([patient_names[i], str(info_W[i, 0]),
                          str(info_W[i, 1]), str(info_W[i, 2]), '%.6f'
                          % (info_W[i, 1] / info_W[i, 0]), '%.6f'
                          % (info_W[i, 2] / info_W[i, 0])])
    info_F_result.append([patient_names[i], str(info_F[i, 0]),
                          str(info_F[i, 1]), str(info_F[i, 2]), '%.6f'
                          % (info_F[i, 1] / info_F[i, 0]), '%.6f'
                          % (info_F[i, 2] / info_F[i, 0])])

# save_matrix_data(info_W_result, file_name=(new_file_name_W +
#                                            'info'), print_=True)
save_matrix_data(info_F_result, file_name=(new_file_name_F +
                                           'info'), print_=True)

[test_patient, train_patient] = generate_dataset()
full_preprocessing(train_patient)

print('Parsing process has finished successfully.')

# EOF

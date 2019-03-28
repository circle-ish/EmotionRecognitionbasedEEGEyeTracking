import os
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
import auxiliary
import params_n_lists

def load_data(dataset_name): 
    if dataset_name == 'seed':
        eegDir = seed_dir
    elif dataset_name == 'moritz':
        eegDir = moritz_dir[0]
        eyeDir = moritz_dir[1]
    elif dataset_name == 'stanford':
        eegDir = stanford_dir[0]
        eyeDir = stanford_dir[1]
    else:
        print_error("load_data: Unknown dataset.")

    fileNames = os.listdir(eegDir)
    fileNames.sort()

    if verbosity_level > 2:
        print(fileNames)

    # loading all data into memory
    eeg_data_list = []
    eye_data_list = []

    for item in fileNames:
        eeg_data = sio.loadmat(eegDir + item)
        eeg_data_list.append(eeg_data)
    
    if dataset_name != 'seed':
        for item in fileNames:
            eye_data = sio.loadmat(eyeDir + item)
            eye_data_list.append(eye_data)

    return [fileNames, 
            eeg_data_list, 
            eye_data_list]

def pick_session(
        dataset_name,
        index,
        eeg_data_list, 
        eye_data_list):

    eeg_used = eeg_data_list[index]
    if dataset_name != 'seed':
        eye_used = eye_data_list[index]
    
    # separate DE-delta features
    if dataset_name == 'seed':
        eeg_train_origin = eeg_used['train_inst_eeg']
        eeg_test_origin = eeg_used['test_inst_eeg']
        eeg_train_alpha = eeg_train_origin[:,62*2:62*3]
        eeg_test_alpha  = eeg_test_origin[:,62*2:62*3]
        eye_train_origin = eeg_used['train_inst_eye']
        eye_test_origin = eeg_used['test_inst_eye']
        train_label = eeg_used['train_label']
        test_label = eeg_used['test_label']
    elif dataset_name == 'moritz'
        eeg_train_origin = eeg_used['cell']['de_data_train'][0][0]
        eeg_test_origin = eeg_used['cell']['de_data_test'][0][0]
        eeg_train_alpha = eeg_train_origin[:,62*2:62*3]
        eeg_test_alpha  = eeg_test_origin[:,62*2:62*3]
        eye_train_origin = eye_used['cell']['eye_data_train'][0][0]
        eye_test_origin = eye_used['cell']['eye_data_test'][0][0]
        train_label = eeg_used['cell']['de_label_train'][0][0]
        test_label = eeg_used['cell']['de_label_test'][0][0]
    else:
        eeg_train_origin = eeg_used['cells']['de_data_train'][0][0]
        eeg_test_origin = eeg_used['cells']['de_data_test'][0][0]
        eeg_train_alpha = eeg_train_origin[:,62*2:62*3]
        eeg_test_alpha  = eeg_test_origin[:,62*2:62*3]
        eye_train_origin = eye_used['cell']['eye_data_train'][0][0]
        eye_test_origin = eye_used['cell']['eye_data_test'][0][0]
        train_label = eeg_used['cells']['de_label_train'][0][0]
        test_label = eeg_used['cells']['de_label_test'][0][0]
    
    # normalize
    scaler = MinMaxScaler()
    eeg_train = scaler.fit_transform(eeg_train_alpha)
    eeg_test = scaler.fit_transform(eeg_test_alpha)
    eye_train = scaler.fit_transform(eye_train_origin)
    eye_test = scaler.fit_transform(eye_test_origin)
    
    train_dict = {'eeg': eeg_train, 'eye': eye_train, 'label': train_label}
    test_dict = {'eeg': eeg_test, 'eye': eye_test, 'label': test_label}
 
    return [train_dict, test_dict]

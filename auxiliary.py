import os, sys
import numpy as np
#import psutil

# function for converting labels into hot encoding for our tensorflow compatibility
def one_hot_encoding(label):
    labels_ohe = np.zeros((label.size,3))
    for index, element in enumerate(label):
        labels_ohe[index][element[0]] = 1
    return labels_ohe

def print_result(name):
    directory = name 
    npsavetxt(directory, np.transpose([dataset_list,
                                   name_list,
                                   epochs_mm_list,
                                   hidden_layer_mm_list,
                                   act_func_mm_list,
                                   hidden_layer_num_list,
                                   hidden_layer_units_list,
                                   training_epochs_list,
                                   init_weight_func_hidden_list,
                                   init_bias_func_hidden_list,
                                   init_weight_func_output_list,
                                   init_bias_func_output_list,
                                   act_func_hidden_list,
                                   dropout_keep_rate_list,
                                   initial_learning_rate_list,
                                   seed_list,
                                   svm_eeg_list,
                                   svm_eye_list,
                                   svm_mm_list,
                                   acc_nn_list
                                   ]))
def npsavetxt(fl, data):
    #proc = psutil.Process()
    #for f in proc.open_files():
    #    print(f.path)
    #print('\t\t',proc.num_fds())
    with open(fl, 'wb') as f:
        print("write", fl)
        try:
            np.savetxt(f, data,delimiter=",", fmt="%s")
        finally:
            f.close()
    return

def nploadtxt(fl):
    ret = 0
    with open(fl, 'rb') as f:
        try:
            ret = np.load(f, delimitier=",")
        finally:
            f.close
    return ret

def npappendtxt(fl, data):
    #proc = psutil.Process()
    #for f in proc.open_files():
    #    print(f.path)
    #print('\t\t',proc.num_fds())
    with open(fl, 'ab') as f:
        try:
            np.savetxt(f, data,delimiter=",", fmt="%s")
        finally:
            f.close()
    return

def usage(progname):
    print('USAGE:')
    print('\t',progname, '-i <source directory> -o <result directory> -t <time directory>')
    print()
    print('Where <source directory> is the folder which contains <ID>_<session nr>_blink.txt, <ID>_<session nr>_sacc.txt, <ID>_<session nr>_fix.txt files with <session nr> being an integer. The <time directory> contains <session nr>_<start/end>.txt files.')
    sys.exit(2)

def print_error(message):
    print("ERROR:")
    print(message)
    sys.exit(2)


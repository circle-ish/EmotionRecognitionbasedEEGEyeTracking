import sys, getopt
import numpy as np
from functools import partial
import math
import tensorflow as tf

#parts
import params_n_lists
import load_n_preprocess
import mf_n_svm
from neural_net import NN
import auxiliary

#funcs
stdv = 2.0
part_func_rn = partial(tf.random_normal, seed=seed)
part_func_tn_stddev = partial(tf.truncated_normal, stddev=math.sqrt(stdv), seed=seed)
part_func_tn_p_one = partial(tf.truncated_normal, stddev=0.1, seed=seed)
part_func_const_p_one = partial(tf.constant, 0.1, dtype=tf.float32)
part_func_const_zero = partial(tf.constant, 0, dtype=tf.float32)
act_tanh = tf.nn.tanh
act_sigmoid = tf.nn.sigmoid
act_relu = tf.nn.relu

def main(
        datasets = ['SEED', 'moritz', 'stanford'], 
        phases = ['train', 'test'], 
        participant = '',
        epochs = [15, 20, 50, 100],
        hidden_units = [60, 100, 140, 400],
        acts = ['sigmoid', 'tanh', 'relu'],
        hidden_layer_num_temp = [2,3],
        hidden_layer_units_temp = [500, 700, 1100],
        training_epochs_temp =  [75, 125, 400],
        init_weight_func_hidden_temp = [part_func_rn, part_func_tn_stddev, part_func_tn_p_one, part_func_const_p_one, part_func_const_zero],
        init_bias_func_hidden_temp = [part_func_const_p_one]
        init_weight_func_output_temp = [part_func_rn]
        init_bias_func_output_temp = [part_func_const_p_one]
        act_func_hidden_temp = [act_tanh, act_sigmoid, act_relu],
        dropout_keep_rate_temp = [0.9],
        initial_learning_rate_temp = [0.001],
        seed_temp = [1, 8888, 1234582, 899953, 958472111, 3234, 832, 23, 99887234, 1239141]):

    all_data = {}
    for dataset in datasets: 
        [fileNames, eeg_data_list, eye_data_list] = load_data(dataset)
        all_data[dataset] = {'fileNames': fileNames, 'eeg_data': eeg_data_list, 'eye_data': eye_data_list}
        for p in phases:
            for it in range(len(eeg_data_list)):
                #current participant:
                name = fileNames[it]
                #choosing participant
                if participant != '':
                    if not name.startswith(participant):
                        continue
                    if verbosity_level > 2: 
                        print('Jumped to participant' + participant)
                else:
                    participant = os.path.splitext(filename)[0].split('.')[0]
                if verbosity_level > 1:
                    print('\n###########################################')
                    print('Starting with participant {}: {}\n'.format(it, participant))

                [train_dict, test_dict] = pick_session(dataset, it, eeg_data_list,  eye_data_list)
                if p == 'train':
                    #[svm_eeg, svm_eye] = svm1(train_dict, test_dict)
        
                #After SVM: Change labels to one-hot-encoding for TF-compatibility
                train_dict['label_tf'] = one_hot_encoding(train_dict['label'])
                test_dict['label_tf'] = one_hot_encoding(test_dict['label'])
                
                #session specific parameters:
                #(suitable_mm, params) = suitable_params[name.split('.')[0]]
                #epochs = [suitable_mm[0]] 
                #hidden_units = [suitable_mm[1]] 
                #acts = [suitable_mm[2]]

                count_mm = 0
                for ei, this_epoch in enumerate(epochs): 
                    for hi, this_hidden in enumerate(hidden_units):
                        for act in acts: #['sigmoid', 'tanh', 'relu']:
                            count_mm += 1
                            [train_features, test_features] = mfnn(ei, this_epoch, hi, this_hidden, act, count_mm, eeg_train, eye_train, eeg_test, eye_test)
                            svm_mm = svm2(train_features, test_features, train_label, test_label)
                            #NN
                            for e in  
                            neurons_firing(dataset, name, this_epoch, this_hidden, seed, act, svm_eeg, svm_eye, svm_mm, train_features, train_label_tf, test_features, test_label_tf) #, params)
    
                    K.clear_session()

def neurons_firing(dataset, name, this_epoch, this_hidden, seed, act, svm_eeg, svm_eye, svm_mm, train_features, train_label_tf, test_features, test_label_tf): #, params):
 #initialising nn
    for seed in seed_temp:
        if seed == 0:
            rng = np.random.RandomState()
            seed = rng.randint(2337203685477580)

        for hidden_layer_num in hidden_layer_num_temp:
            for hidden_layer_units in hidden_layer_units_temp:
                hidden_layer_size = [hidden_layer_units] * hidden_layer_num
                for training_epochs in training_epochs_temp:
                    for init_weight_func_hidden in init_weight_func_hidden_temp:
                        func_in_question = init_weight_func_hidden 
                        if func_in_question == part_func_rn or 
                                func_in_question == part_func_tn_stddev or
                                func_in_question == part_func_tn_p_one:
                            init_weight_func_hidden = partial(func_in_question, seed=seed)
                        
                        for init_bias_func_hidden in init_bias_func_hidden_temp:
                            for init_weight_func_output in init_weight_func_output_temp:
                                func_in_question = init_bias_func_hidden 
                                if func_in_question == part_func_rn or 
                                        func_in_question == part_func_tn_stddev or
                                        func_in_question == part_func_tn_p_one:
                                    init_bias_func_hidden = partial(func_in_question, seed=seed)

                                for init_bias_func_output in init_bias_func_output_temp:
                                    for act_func_hidden in act_func_hidden_temp:
                                        for dropout_keep_rate in dropout_keep_rate_temp:
                                            for initial_learning_rate in initial_learning_rate_temp:
                                                nn = NN(
                                                        data_dim = this_hidden,
                                                        hidden_layer_size = hidden_layer_size,
                                                        init_weight_func_hidden = init_weight_func_hidden,
                                                        init_bias_func_hidden = init_bias_func_hidden,
                                                        init_weight_func_output = init_weight_func_output,
                                                        init_bias_func_output = init_bias_func_output,
                                                        activation_func_hidden = act_func_hidden,
                                                        dropout_keep_rate = dropout_keep_rate,            
                                                        initial_learning_rate = initial_learning_rate,
                                                        training_logging_interval = 10000
                                                        )

                                                if p == 'train':
                                                    for epo in range(training_epochs):
                                                        nn.fit(train_features, train_label_tf)
                                                if p == 'test':
                                                    nn.test(test_features, test_label_tf)
        
                                                # get accuracies
                                                acc_nn = nn.accuracy_out
                                                
                                                # store SVM and NN-results in lists
                                                dataset_list.append(dataset)
                                                name_list.append(os.path.splitext(name)[0])
                                                epochs_mm_list.append(this_epoch)
                                                hidden_layer_mm_list.append(this_hidden)
                                                act_func_mm_list.append(act)
                                                hidden_layer_num_list.append(hidden_layer_num)
                                                hidden_layer_units_list.append(hidden_layer_units)
                                                training_epochs_list.append(training_epochs)
                                                stdv_list.append([stdv])
                                                init_weight_func_hidden_list.append(init_weight_func_hidden.func.__name__)
                                                init_bias_func_hidden_list.append(init_bias_func_hidden.func.__name__)
                                                init_weight_func_output_list.append(init_weight_func_output.func.__name__)
                                                init_bias_func_output_list.append(init_bias_func_output.func.__name__)
                                                act_func_hidden_list.append(act_func_hidden.__name__) 
                                                dropout_keep_rate_list.append(dropout_keep_rate)
                                                initial_learning_rate_list.append(initial_learning_rate)
                                                seed_list.append(seed)
                                                svm_eeg_list.append(svm_eeg) 
                                                svm_eye_list.append(svm_eye)
                                                svm_mm_list.append(svm_mm)
                                                acc_nn_list.append(acc_nn)
        
                                                nn.close()
                                                tf.reset_default_graph() ####################################################


if __name__ == "__main__":
    #check commandline args
    progname = sys.argv[0]
    if len(argv) != prog_arg_number:
        usage(progname)
    try:
        opts, args = getopt.getopt(argv[1:],"hd:p:t:")
    except getopt.GetoptError:
        usage(progname)
    
    params = {}
    for opt, arg in opts:
        if opt == '-h':
            usage(progname)
        #dataset - seed, moritz, stanford; separated by ',' without space
        elif opt in ("-d"):
            datasets = arg.split(',')
            for d in datasets:
                if d != 'seed' and d != 'moritz' and d != 'stanford':
                    print_error("Dataset " + d + " does not exist.")
            params['datasets'] = datasets
        #participant - as file starts
        elif opt in ("-p"):
            participant = arg
            params['participant'] = participant
        #phase - test, training; separated by ',' without space
        elif opt in ("-t"):
            phases = arg.split(',')
            for p in phases:
                if p != 'train' and p != 'test':
                    print_error("Phase " + p + " unknown.")
            params['phases'] = phases
        else: print_error("Unknown parameter option "+opt+".")

    main(params)

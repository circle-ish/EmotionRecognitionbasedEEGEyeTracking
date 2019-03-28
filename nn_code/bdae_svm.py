import sys
import os
import numpy as np
import scipy.io as sio
from keras import backend as K
from keras.layers import Dense, Activation, Input, merge, Lambda
from keras.models import Model
from keras.optimizers import SGD,Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import BernoulliRBM
import keras.backend.tensorflow_backend as KTF
from keras import backend as K
import tensorflow as tf
sys.path.append(os.path.join(os.path.join(os.path.join(os.path.dirname(os.path.abspath(os.path.curdir)),'3rdParty'),'libsvm'),'python'))
from svmutil import *
from keras.regularizers import l2

def create_random_lr_hs(lr,hs1,hs2,total_size):
    lr.sort()
    hs1.sort()
    hs2.sort()
    random_value = np.random.rand(total_size,3)
    random_value[:,0] = random_value[:,0]*(lr[1]-lr[0])+lr[0]
    random_value[:,1] = random_value[:,1]*(hs1[1]-hs1[0])+hs1[0]
    random_value[:,2] = random_value[:,2]*(hs2[1]-hs2[0])+hs2[0]

    return random_value
def get_session(gpu_fraction=0.3):
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    if num_threads:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def expand_label(label):
    label_exp = np.zeros([label.shape[0],3])
    for i,l in enumerate(label):
        label_exp[i,l+1] = 1
    return label_exp

eegDir = '../Data/EEG_split/'
eyeDir = '../Data/Eye_split/'
total_size=25
learning_rate = [-5,-1]
learning_rate2 = [-7,-3]
l2_regularizer = [-7,-2]

fileNames = os.listdir(eegDir)
fileNames.sort()
tmp_filelist = []
print(fileNames)

# loading all data into memory
eeg_data_list = []
eye_data_list = []
for item in fileNames:
    if (item[-4:] != '.mat'):
        continue
    eeg_data = sio.loadmat(eegDir + item)
    eye_data = sio.loadmat(eyeDir + item)
    eeg_data_list.append(eeg_data)
    eye_data_list.append(eye_data)
    tmp_filelist.append(item)

fileNames = tmp_filelist

result = np.zeros([len(eeg_data_list),total_size])
record = np.zeros([len(eeg_data_list)*total_size,9])
for fn in range(len(eeg_data_list)):
    label_array = np.zeros((343,))
    eeg_train_features = []
    eeg_test_features = []

    eeg_train_features_below = []
    eeg_test_features_below = []

    eeg_train_features_above = []
    eeg_test_features_above = []

    scaler = MinMaxScaler()
    eeg_used = eeg_data_list[fn]
    eye_used = eye_data_list[fn]
    # separate DE-delta features
    eeg_train_origin = eeg_used['train_de']
    eeg_test_origin = eeg_used['test_de']
    eeg_train_alpha = eeg_train_origin
    eeg_test_alpha  = eeg_test_origin

    eye_train_origin = eye_used['train_data_eye']
    eye_test_origin = eye_used['test_data_eye']
    train_label = expand_label(eye_used['train_label_eye'])
    test_label = expand_label(eye_used['test_label_eye'])

    # normalize
    eeg_train = scaler.fit_transform(eeg_train_alpha)
    eeg_test = scaler.fit_transform(eeg_test_alpha)
    eye_train = scaler.fit_transform(eye_train_origin)
    eye_test = scaler.fit_transform(eye_test_origin)
    # network parameters
    x_row, x_col = eeg_train.shape
    y_row, y_col = eye_train.shape
    del eeg_used, eye_used
    del eeg_train_origin, eeg_test_origin, eye_train_origin, eye_test_origin,eeg_train_alpha, eeg_test_alpha
    del scaler

    for s in range(total_size):
        #random number
        lr = np.random.rand()*(learning_rate[1]-learning_rate[0])+learning_rate[0]
        lr = 10**lr
        lr2 = np.random.rand()*(learning_rate2[1]-learning_rate2[0])+learning_rate2[0]
        lr2 = 10**lr2
        eeg_hs = np.random.randint(int(x_col/3),x_col)
        eye_hs = np.random.randint(int(y_col/3),y_col)
        merge_hs = eeg_hs+eye_hs
        hs1 = np.random.randint(int(merge_hs/3),merge_hs)
        hs2 = np.random.randint(int(hs1/3),hs1)
        l2_reg = np.random.rand()*(l2_regularizer[1]-l2_regularizer[0])+l2_regularizer[0]
        l2_reg = 10**l2_reg

        # RBM initialize
        rbm1 = BernoulliRBM(n_components=eeg_hs, batch_size=100,n_iter=20)
        rbm2 = BernoulliRBM(n_components=eye_hs,batch_size=100, n_iter=20)
        rbm3 = BernoulliRBM(n_components=hs1,batch_size=100, n_iter=20)
        rbm4 = BernoulliRBM(n_components=hs2,batch_size=100, n_iter=20)
        print('RBM 1 training\n')
        rbm1.fit(eeg_train)
        hidden_eeg = rbm1.transform(eeg_train)
        weights_eeg = rbm1.components_
        print('RBM 2 training\n')
        rbm2.fit(eye_train)
        hidden_eye = rbm2.transform(eye_train)
        weights_eye = rbm2.components_
        print('RBM 3 training \n')
        conca_data = np.append(hidden_eeg, hidden_eye, axis=1)
        rbm3.fit(conca_data)
        hidden_merge = rbm3.transform(conca_data)
        weights_merge = rbm3.components_
        print('RBM 4 training \n')
        rbm4.fit(hidden_merge)
        weights_fea = rbm4.components_

        # derAE netwrok structure
        KTF.set_session(get_session())
        def get_eeg_part(nparray):
            global eeg_hs
            return nparray[:,:eeg_hs]
        def get_eye_part(nparray):
            global eeg_hs
            return nparray[:, eeg_hs:]
        print('Model structure begin ... \n')
        print('Input layers ...\n')
        x_input = Input(shape=(x_col,), name='x_input')
        y_input = Input(shape=(y_col,), name='y_input')
        print('hidden layers ...\n')
        x_hidden = Dense(eeg_hs, weights=[weights_eeg.T, rbm1.intercept_hidden_], activation='sigmoid',name='x_hidden')(x_input)
        y_hidden = Dense(eye_hs, weights=[weights_eye.T, rbm2.intercept_hidden_], activation='sigmoid',name='y_hidden')(y_input)
        print('merge layers ... \n')
        merge_xy = merge([x_hidden, y_hidden], mode='concat')
        merge_layer_2 = Dense(hs1, weights=[weights_merge.T, rbm3.intercept_hidden_], activation='sigmoid', name='merged')(merge_xy)
        feature_layer = Dense(hs2, weights=[weights_fea.T, rbm4.intercept_hidden_], activation='sigmoid', name='merged_layer_2')(merge_layer_2)
        # decoding
        print('decoding processing \n')
        merge_layer_2_t = Dense(hs1, weights=[weights_fea, rbm4.intercept_visible_], activation='sigmoid',name='merge_layer_2_t')(feature_layer)
        merge_xy_t = Dense(merge_hs,weights=[weights_merge, rbm3.intercept_visible_], activation='sigmoid',name='merge_t')(merge_layer_2_t)
        x_hidden_t = Lambda(get_eeg_part, output_shape=(eeg_hs,))(merge_xy_t)
        y_hidden_t = Lambda(get_eye_part, output_shape=(eye_hs,))(merge_xy_t)
        x_recon = Dense(x_col, weights=[weights_eeg, rbm1.intercept_visible_], activation='sigmoid',name='x_recon')(x_hidden_t)
        y_recon = Dense(y_col, weights=[weights_eye, rbm2.intercept_visible_], activation='sigmoid',name='y_recon')(y_hidden_t)

        model = Model(input=[x_input, y_input], output=[x_recon, y_recon])
        adam = Adam(lr=lr)
        model.compile(optimizer=adam,loss='mean_squared_error')
        model.fit([eeg_train, eye_train],[eeg_train, eye_train], nb_epoch=500, batch_size=100)

        adam.lr = lr2
        labels = Dense(3,name='svm',W_regularizer=l2(l2_reg))(feature_layer)
        model2 = Model(input=[x_input, y_input], output=[labels])
        model2.compile(optimizer=adam,loss='hinge')
        for i in range(7):
            model2.layers[i].set_weights(model.layers[i].get_weights())
        model2.fit([eeg_train,eye_train],train_label,nb_epoch=1000,batch_size=100)

        feature_res = K.function([model2.layers[0].input, model2.layers[1].input],[model2.layers[6].output]) # this is the middle layer
        feature_res_below = K.function([model2.layers[0].input, model2.layers[1].input],[model2.layers[5].output]) # this is the layer below the middle layer

        ## get the extracted feature
        train_features = feature_res([eeg_train, eye_train])[0]
        test_features = feature_res([eeg_test, eye_test])[0]
        eeg_train_features.append(train_features.tolist())
        eeg_test_features.append(test_features.tolist())

        train_features_below = feature_res_below([eeg_train, eye_train])[0]
        test_features_below = feature_res_below([eeg_test, eye_test])[0]
        eeg_train_features_below.append(train_features_below.tolist())
        eeg_test_features_below.append(test_features_below.tolist())

        predict_label = model2.predict([eeg_test,eye_test])
        p_acc = np.average(np.argmax(predict_label,axis=1) == np.argmax(test_label,axis=1))

        record[fn*total_size+s,:] = np.array([fn,lr,lr2,eeg_hs,eye_hs,hs1,hs2,l2_reg,p_acc])
        result[fn,s] = p_acc

        del rbm1, hidden_eeg, weights_eeg
        del rbm2, hidden_eye, weights_eye
        del rbm3, weights_merge, conca_data
        del hidden_merge, rbm4, weights_fea

        K.clear_session()

    np.save('./de_all_res/train_features/'+fileNames[fn],eeg_train_features)
    del eeg_train_features
    np.save('./de_all_res/test_features/'+fileNames[fn],eeg_test_features)
    del eeg_test_features

    np.save('./de_all_res/train_features_below/'+fileNames[fn],eeg_train_features_below)
    del eeg_train_features_below
    np.save('./de_all_res/test_features_below/'+fileNames[fn],eeg_test_features_below)
    del eeg_test_features_below

np.savetxt('./de_all_res/accuracy/result1.csv',result,delimiter=',')
np.savetxt('./de_all_res/accuracy/record1.csv',record,delimiter=',')

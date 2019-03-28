from keras import backend as K
from keras.layers import Dense, Activation, Input, merge, Lambda
from keras.models import Model
from keras.optimizers import SGD
from sklearn.neural_network import BernoulliRBM
from sklearn import svm

def svm1(eeg_train, eye_train, eeg_test, eye_test, train_label, test_label):
    # SVM classification with single modalities
    clf_eeg = svm.SVC()
    clf_eeg.fit(eeg_train, np.ravel(train_label))
    svm_eeg = clf_eeg.score(eeg_test,np.ravel(test_label))
    clf_eye = svm.SVC()
    clf_eye.fit(eye_train, np.ravel(train_label))
    svm_eye = clf_eye.score(eye_test, np.ravel(test_label))
    return [svm_eeg, svm_eye] 

def svm2(train_features, test_features, train_label, test_label):
    # SVM classification with multiple modalities
    clf = svm.SVC()
    clf.fit(train_features, np.ravel(train_label))
    svm_mm = clf.score(test_features, np.ravel(test_label))
    return svm_mm

def mfnn(ei, this_epoch, hi, this_hidden, act, count_mm, eeg_train, eye_train, eeg_test, eye_test):
     x_row, x_col = eeg_train.shape
     y_row, y_col = eye_train.shape       
     
     res_row = ei * 10 + hi
     # RBM initialize
     rbm1 = BernoulliRBM(n_components=this_hidden, batch_size=100,n_iter=20)
     rbm2 = BernoulliRBM(n_components=this_hidden,batch_size=100, n_iter=20)
     rbm3 = BernoulliRBM(n_components=this_hidden,batch_size=100, n_iter=20)
     # rbm1
     #print('RBM 1 training\n')
     rbm1.fit(eeg_train)
     hidden_eeg = rbm1.transform(eeg_train)
     weights_eeg = rbm1.components_
     # rbm2
     #print('RBM 2 training\n')
     rbm2.fit(eye_train)
     hidden_eye = rbm2.transform(eye_train)
     weights_eye = rbm2.components_
     # rbm3
     #print('RBM 3 training \n')
     #print('RBM 3 training \n')
     #print('RBM 3 training \n')
     conca_data = np.append(hidden_eeg, hidden_eye, axis=1)
     rbm3.fit(conca_data)
     weights_merge = rbm3.components_

     # network structure
     kparams = {'this_hidden':this_hidden}
     setattr(K, 'kparams', kparams)

     def get_eeg_part(nparray):
         return nparray[:,:K.kparams['this_hidden']]
     def get_eye_part(nparray):
         return nparray[:, K.kparams['this_hidden']:]

     #print('Model structure begin ... \n')
     #print('Input layers ...\n')
     x_input = Input(shape=(x_col,), name='x_input')
     y_input = Input(shape=(y_col,), name='y_input')

     #print('hidden layers ...\n')
     x_hidden = Dense(this_hidden, weights=[weights_eeg.T, rbm1.intercept_hidden_], activation=act,name='x_hidden')(x_input)
     y_hidden = Dense(this_hidden, weights=[weights_eye.T, rbm2.intercept_hidden_], activation=act,name='y_hidden')(y_input)

     #print('merge layers ... \n')
     merge_xy = merge([x_hidden, y_hidden], mode='concat')
     feature_layer = Dense(this_hidden, weights=[weights_merge.T, rbm3.intercept_hidden_], activation=act, name='merged')(merge_xy)
     # decoding
     #print('decoding processing \n')
     merge_xy_t = Dense(2*this_hidden,weights=[weights_merge, rbm3.intercept_visible_], activation=act,name='merge_t')(feature_layer)
     x_hidden_t = Lambda(get_eeg_part, output_shape=(this_hidden,))(merge_xy_t)
     y_hidden_t = Lambda(get_eye_part, output_shape=(this_hidden,))(merge_xy_t)

     x_recon = Dense(x_col, weights=[weights_eeg, rbm1.intercept_visible_], activation=act,name='x_recon')(x_hidden_t)
     y_recon = Dense(y_col, weights=[weights_eye, rbm2.intercept_visible_], activation=act,name='y_recon')(y_hidden_t)

     model = Model(input=[x_input, y_input], output=[x_recon, y_recon])
     model.compile(optimizer='rmsprop',loss='mean_squared_error')
     model.fit([eeg_train, eye_train],[eeg_train, eye_train], nb_epoch=this_epoch, batch_size=100, verbose=0)
     #print('\n extracting middle features\n')

     feature_res = K.function([model.layers[0].input, model.layers[1].input],[model.layers[5].output])

     ## get the extracted feature
     train_features = feature_res([eeg_train, eye_train])[0]
     test_features = feature_res([eeg_test, eye_test])[0]
     print('Finished {} modality fusion for this participant'.format(count_mm))
     # extract train and test features for each configuration
     #np.save(os.path.splitext(name)[0] + '_train_features_' + str(this_epoch) + '_' + str(this_hidden), train_features)                    
     #np.save(os.path.splitext(name)[0] + '_test_features_' + str(this_epoch) + '_' + str(this_hidden), test_features)
     return [train_features, test_features]


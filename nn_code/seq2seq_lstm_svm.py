"""
seq2one lstm model with one layer and 128 hidden units.
on the top of lstm is one-layer-softmax
"""

from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent,LSTM,Merge
from keras.layers.core import Dropout
from keras.optimizers import Adam
from keras.regularizers import l2

def get_model(input_dim1=310,input_dim2=41,hidden_size1=128,hidden_size2=32,max_len=60,layers=1,verbose = False,learning_rate=0.001,dropout=True,dropout_prob=0.5,l2_reg=0.01):
    EMOTION_CLASS_NUM=3
    if verbose:
        print('Build model...')
    model_lstm1 = Sequential()
    model_lstm1.add(LSTM(hidden_size1,input_shape=(max_len,input_dim1),return_sequences = True,consume_less='gpu'))
    if dropout:
        model_lstm1.add(TimeDistributed(Dropout(dropout_prob)))
    model_lstm2 = Sequential()
    model_lstm2.add(LSTM(hidden_size2,input_shape=(max_len,input_dim2),return_sequences=True,consume_less='gpu'))
    if dropout:
        model_lstm2.add(TimeDistributed(Dropout(dropout_prob)))

    merge = Merge([model_lstm1,model_lstm2],mode='concat')
    model = Sequential()
    model.add(merge)
    model.add(TimeDistributed(Dense(EMOTION_CLASS_NUM,W_regularizer=l2(l2_reg))))
    adam = Adam(lr = learning_rate)
    model.compile(loss='hinge',optimizer=adam,metrics=['accuracy'])
    return model
if __name__ == '__main__':
    model = (get_model())
    model.summary()

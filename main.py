import pickle
import re
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

import unidecode


def split_words(txt):
    w12 = re.sub('[\[\]\.,:;!\?\(\)=\-\+\"\'—ªº”“\*\d\n\r\t]', ' ', txt)
    # w12 = list(filter(None, w12))
    return w12


def vector_of(w):
    if w[0] == ' ': w = w[1:]
    if w in wdic:
        return wdic[w]
    return None


def vector_of_seq(seq):
    vs = []
    for s in seq:
        v = vector_of(s)
        if v == None :
            return None
        vs.append(v)
    return vs

filename = "sertoes.txt"
raw_text = open(filename).read().lower()
# raw_text = unidecode.unidecode(raw_text)

raw_text = split_words(raw_text)
#print(raw_text)
word_list = re.split(' ', raw_text)
word_list = list(filter(None, word_list))
# print( word_list)


all_wdata = pickle.load(open("wordvector.p", "rb"))
wdic = {item: index for index, item, vector in all_wdata}
#wvector = {index: vector for index, item, vector in all_wdata}


vector_lenght = len(all_wdata[0][2])

seq_length = 8
dataX = []
dataY = []
n_words = len(word_list)
for i in range(0, n_words - seq_length, 1):
    seq_in = word_list[i:i + seq_length]
    seq_out = word_list[i + seq_length]

    v_out = vector_of(seq_out)
    if (v_out ==None ):
        i = i+seq_length
        continue
    v_in = vector_of_seq(seq_in)
    if (v_in ==None ):
        continue

    #print(seq_in , seq_out)
    #print(v_in, v_out)
    dataX.append(np.array([ np.array( (all_wdata[j][2])) for j in v_in]))
    dataY.append(np.array(all_wdata[v_out][2]))

    if (len(dataY)> 900000):
        break
    #if i%100  == 0 :
    #    print(len(dataY[-1]))
        #print(dataX[-1], dataY[-1])
        #print(dataX[-1]+dataX[-1])

n_patterns = len(dataX)

print("Total Patterns: ", n_patterns)

X = np.reshape(dataX, (n_patterns, seq_length, vector_lenght))
Y = np.reshape(dataY, (n_patterns, vector_lenght))


# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(seq_length, vector_lenght)))
model.add(Dropout(0.1))
model.add(Dense(Y.shape[1], activation='relu'))
model.add(Dense(Y.shape[1], activation='relu'))
model.compile(loss='mean_absolute_error', optimizer='adam')


# define the checkpoint
filepath="weights-improvement-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(X, Y, epochs=100, batch_size=256, callbacks=callbacks_list)

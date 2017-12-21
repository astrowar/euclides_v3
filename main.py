import pickle
import re
import numpy
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

    print(seq_in , seq_out)
    print(v_in, v_out)
    #dataX.append([char_to_int[char] for char in seq_in])
    #dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

exit(0)
for w in word_list:
    if w[0] == ' ': w = w[1:]
    if w in wdic:
        continue
    print("unknoun word '", w, "'")

# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(10, 50)))
model.add(Dropout(0.2))
model.add(Dense(20, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

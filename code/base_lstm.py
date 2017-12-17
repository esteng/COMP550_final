import argparse 
import numpy as np 
import math

from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Flatten
from keras.layers.wrappers import Bidirectional, TimeDistributed
# from keras.regularizers import l2


np.random.seed(100)

def define_model(embedding_size, tag_size, input_length, embedding = False, vocab_dim = 0):
    model = Sequential()
    if embedding:
        print "defining with embedding layer"
        # maybe change to 300
        model.add(Embedding(vocab_dim, 128, input_length=input_length))
        model.add(Bidirectional(LSTM(128, return_sequences=True),input_shape=(input_length,embedding_size)))
    else:
        print "defining no embedding layer"
        model.add(Bidirectional(LSTM(128, return_sequences=True),input_shape=(input_length,embedding_size)))
                               
    model.add(Dropout(0.5))
    model.add(Dense(tag_size, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    return model










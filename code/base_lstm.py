import argparse 
import numpy as np 
import math

from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Flatten
from keras.layers.wrappers import Bidirectional, TimeDistributed

np.random.seed(100)

def define_model(embedding_size, tag_size, input_length, layer_number, embedding = False, vocab_dim = 0):
    model = Sequential()
    if embedding:
        # maybe change to 300
        model.add(Embedding(vocab_dim, 300, input_length=input_length))
    model.add(Bidirectional(LSTM(128, return_sequences=True),input_shape=(input_length,embedding_size)))
    
    if layer_number > 1:
        i = 1
        while i < layer_number:
            model.add(Dropout(0.5))
            model.add(Bidirectional(LSTM(128, return_sequences=True),input_shape=(input_length,embedding_size)))
            i+=1

    model.add(Dropout(0.5))
    model.add(Dense(tag_size, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    return model










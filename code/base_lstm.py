from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.regularizers import l2


from data import process_csv, process_file, make_x_y
import numpy as np 
import math




np.random.seed(100)

def define_model(embedding_size, tag_size):
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True,
                                        W_regularizer=l2(0.000), \
                                        U_regularizer=l2(0.000)), 
                                                    input_shape=(29,300)))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(10, activation='softmax',W_regularizer=l2(0.000), \
                                        b_regularizer=l2(0.000))))

    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    return model


print("loading data")
# data = process_csv("../data/entity-annotated-corpus/ner_dataset.csv")
data = process_file("../data/news_tagged_data.txt")

# print("getting X Y sets")
X, Y, word_embeddings, tag_embeddings = make_x_y(data, 300, True)



train, test, dev = .7, .2, .1

train_split = int(len(X)*train)
test_split = train_split+1+int(math.floor(len(X)*test))


X_train, Y_train = X[:train_split], Y[:train_split]
X_test, Y_test = X[train_split +1 : test_split], Y[train_split + 1:test_split]
X_dev, Y_dev = X[test_split +1 :], Y[test_split +1 :]


print("data shape:")
print("train: {}, test: {}, dev: {}".format(X_train.shape, X_test.shape, X_dev.shape))


# tag_to_vec, vec_to_tag = tag_embeddings
# tag_length = len(tag_to_vec.values())
tag_length = 10
word_embedding_size = 300
print("defining model...")
model = define_model(word_embedding_size, tag_length)
print "model summary:"
print model.summary()
print("training...")
model.fit(X_train, Y_train, epochs=20, batch_size=32, verbose=1,  shuffle=True)
scores = model.evaluate(X_dev, Y_dev, verbose=1)
model.validate(X_dev, Y_dev)






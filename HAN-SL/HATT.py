import numpy as np
import pandas as pd
from collections import defaultdict
import re

from bs4 import BeautifulSoup

import sys
import os
import csv
import json

os.environ['KERAS_BACKEND']='theano'
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import model_from_json
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers

import preprocessor as p
np.random.seed(12)
print(12)
p.set_options(p.OPT.URL, p.OPT.EMOJI,p.OPT.NUMBER,p.OPT.SMILEY)


MAX_SENT_LENGTH = 30
MAX_SENTS = 20000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.25


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

server="/home/rs/jpatro/pushpendra/"
data_train = pd.read_csv(server+"text.csv")
print (data_train.shape)
print(data_train.iloc[0])
from nltk import tokenize


reviews = []
labels = []
texts = []


for idx in range(data_train.content.shape[0]):
    text = BeautifulSoup(data_train.content[idx])
    print(idx)
    text = p.clean(text.get_text().encode('ascii','ignore').decode('utf-8'))
    texts.append(text)
    sentences = tokenize.sent_tokenize(text)
    reviews.append(sentences)

    labels.append(data_train.Tag[idx]-1)

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
#print(texts[:1][:5])


data = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')

for i, sentences in enumerate(reviews):
    for j, sent in enumerate(sentences):
        if j < MAX_SENTS:
            wordTokens = text_to_word_sequence(sent)
            k = 0
            for _, word in enumerate(wordTokens):
                if k < MAX_SENT_LENGTH and tokenizer.word_index[word] < MAX_NB_WORDS:
                    data[i, j, k] = tokenizer.word_index[word]
                    k = k + 1


word_index = tokenizer.word_index
print('Total %s unique tokens.' % len(word_index))

labels = to_categorical(np.asarray(labels))
print(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
#np.save(server+"training_data.npy",data)
#np.save(server+"training_label.npy",labels)
print("done saving")


indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

print("Total_data")
for item in indices:
	print(item)

print("Test Data")
for item in indices[-nb_validation_samples:]:
	print(item)




x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]



print('Number of positive and negative reviews in traing and validation set')
print( y_train.sum(axis=0))
print (y_val.sum(axis=0))



GLOVE_DIR = server+"glove"
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings_index))

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SENT_LENGTH,
                            trainable=True,
                            mask_zero=True)

class AttLayer(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])




sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)
l_lstm = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
l_att = AttLayer(100)(l_lstm)
sentEncoder = Model(sentence_input, l_att)

review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
review_encoder = TimeDistributed(sentEncoder)(review_input)
l_lstm_sent = Bidirectional(GRU(100, return_sequences=True))(review_encoder)
l_att_sent = AttLayer(50)(l_lstm_sent)
preds = Dense(4, activation='softmax')(l_att_sent)
model = Model(review_input, preds)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics= ['mae', 'acc'])

print("model fitting - Hierachical attention network=> rs=10 and withotu class_weight")

##Add checkpoints
# checkpoint
model.summary()
class_weights ={0:0.82958199, 1:0.77945619, 2:0.84039088, 3:3.10843373}
model.fit(x_train, y_train,validation_data=(x_val,y_val),
          nb_epoch=2, batch_size=20, verbose=1)


score=model.evaluate(x_val,y_val)
print(score)
result=[]
for item in y_val:
    print(item)
    result.append(item)
    
prediction=model.predict(x_val)
for item in prediction:
    result.append(item)
    print(item)
with open(server+'HATT_result2_12_1.csv', 'w') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(result)

model_json = model.to_json()
with open(server+"HATT_12_1.json", "w") as json_file:
    json_file.write(model_json)
model.save(server+"HATT_2_12_1.h5")

# serialize weights to HDF5

print("Saved model to disk")



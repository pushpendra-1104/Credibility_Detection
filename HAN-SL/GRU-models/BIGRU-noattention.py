import numpy as np
import pandas as pd
from collections import defaultdict
import re
import csv
from bs4 import BeautifulSoup
import sys
import os
import multiprocessing as mp
os.environ['KERAS_BACKEND']='theano'
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import merge
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers
import preprocessor as p
from nltk import tokenize


##Configuration used for data cleaning and word embeddings vector creation
p.set_options(p.OPT.URL, p.OPT.EMOJI,p.OPT.NUMBER,p.OPT.SMILEY)

MAX_SEQUENCE_LENGTH = 10000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.25

np.random.seed(12)
server="/home/others/15EE10031/pushpendra/"

##Load Socio linguistic features data which consist LIWC,Empath and other linguistic features.
data1=pd.read_csv(server+"features/Empath_features1.csv")
data1=data1.drop(["Tag"],axis=1)
data2=pd.read_csv(server+"features/features11.csv")

#Merge both data and normalize on the basis of number of tweets in a event 
data1=pd.merge(data1, data2, on="Event")
features=list(data1)
features=[item for item in features if item not in ["Event","Tag","tweetcount"]]
for item in features:
    data1[item]=data1[item]/data1["tweetcount"]
data1=data1.drop(["tweetcount"],axis=1)
data1_corr=data1.corr()
print(data1_corr["Tag"])
top_cols=np.where(abs(data1_corr["Tag"])>0.1)[0]
top_cols=top_cols[:-1]
data1=data1.drop(["Event","Tag"],axis=1)
print(data1_corr["Tag"][data1_corr["Tag"]>0.1])
print(len(top_cols),top_cols)
data1=data1.iloc[:,top_cols]
data1=data1.as_matrix()
print(np.shape(data1))


#Load text data for each event and further tokenize it for creating word embedding_vector
data_train = pd.read_csv(server+"text.csv")
print (data_train.shape)
print(data_train.iloc[0])

texts = []
labels = []

##Data cleaning using BeautifulSoup and preprocessing library for tweets
for idx in range(data_train.content.shape[0]):
    print(idx)
    text = BeautifulSoup(data_train.content[idx])
    texts.append(p.clean(text.get_text().encode('ascii','ignore').decode('utf-8')))

    temp=np.zeros(4)
    temp[data_train.Tag[idx]-1]=1
    labels.append(data_train.Tag[idx]-1)


tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(labels))

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)



indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
Liwc_data = data1[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])


###Create word embedding layer from already trained Glove embeddding word vectors
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
    print(i)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)


def create_model():
	sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
	embedded_sequences = embedding_layer(sequence_input)
	l_lstm = Bidirectional(GRU(100))(embedded_sequences)
	otherInp = Input(shape = (Liwc_train.shape[1], ))
	concatenatedFeatures = keras.layers.Concatenate(axis = -1)([l_lstm , otherInp])
	preds = Dense(4, activation='softmax')(concatenatedFeatures)
	model=Model(inputs=[sequence_input,otherInp], outputs=preds)


	model.compile(loss='categorical_crossentropy',
		      optimizer='rmsprop',
		      metrics=['acc'])
	model.summary()
	return model
    
#Fitting the model with 5 cross validations
np.random.seed(12)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
Liwc_data = data1[indices]
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.25, random_state=10)
cv_score=[]
for index, (train_indices, val_indices) in enumerate(sss.split(data,np.zeros(shape=(data.shape[0], 1)))):
    print ("Training on fold " + str(index+1) + "/5...")
    # Generate batches from indices
    x_train, x_val = data[train_indices], data[val_indices]
    y_train, y_val = labels[train_indices], labels[val_indices]
    Liwc_train,Liwc_val = Liwc_data[train_indices],Liwc_data[val_indices]
    print("Liwc data shape",Liwc_train.shape,Liwc_val.shape)

    print('Number of different credibility levels in training and validation set')
    print( y_train.sum(axis=0))
    print (y_val.sum(axis=0))
    # Clear model, and create it
    model = None
    model = create_model()
    


    print("model fitting - BI GRU with no attention")
    model.fit([x_train,Liwc_train], y_train,validation_data=([x_val,Liwc_val],y_val),
              nb_epoch=3, batch_size=20, verbose=1)


    score=model.evaluate([x_val,Liwc_val],y_val)
    print(score)
    cv_score.append(score[1])
    result=[]
    for item in y_val:
        print(item)
        result.append(item)
    
    prediction=model.predict([x_val,Liwc_val])
    for item in prediction:
        result.append(item)
        print(item)
    with open(server+'BIGRU_noatt'+ str(index+1)+'.csv', 'w') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(result)

    model_json = model.to_json()
    with open(server+'BIGRU_noatt'+ str(index+1)+'.json', "w") as json_file:
         json_file.write(model_json)

print(np.mean(cv_score))

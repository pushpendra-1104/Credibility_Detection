import numpy as np
import pandas as pd
from collections import defaultdict
import re
from bs4 import BeautifulSoup
import sys
import os
import csv
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import keras
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model
from sklearn.model_selection import StratifiedKFold
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers
import preprocessor as p
from nltk import tokenize
from sklearn.model_selection import StratifiedShuffleSplit

os.environ['KERAS_BACKEND']='theano'


##Configuration used for data cleaning and word embeddings vector creation
p.set_options(p.OPT.URL, p.OPT.EMOJI,p.OPT.NUMBER,p.OPT.SMILEY)
MAX_SENT_LENGTH = 30
MAX_SENTS =10000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.25

np.random.seed(12)
server="/home/others/15EE10031/pushpendra/"
##Load Socio linguistic features data which consist LIWC,Empath and other linguistic features.
Empath_data=pd.read_csv(server+"features/Empath_features1.csv")
Empath_data=Empath_data.drop(["Tag"],axis=1)
Linguistic_data=pd.read_csv(server+"features/features11.csv")

##Merge both data and normalize on the basis of number of tweets in a event 
data1=pd.merge(Empath_data, Linguistic_data, on="Event")
features=list(data1)
features=[item for item in features if item not in ["Event","Tag","tweetcount"]]
for item in features:
    data1[item]=data1[item]/Empath_data["tweetcount"]
data1=data1.drop(["tweetcount"],axis=1)
def find_correlated_cols(data):
    data1_corr=data.corr()
    print(data1_corr["Tag"])
    print(data1_corr["Tag"][data1_corr["Tag"]>0.1])
    top_cols=np.where(abs(data1_corr["Tag"])>0.1)[0]
    top_cols=top_cols[:-1]

##Extract Linguistic features for data which are highly correlated with event Credbiliy, hreshold can be used we used 0.1
data1=data1.drop(["Event","Tag"],axis=1)
top_cols=find_correlated_cols(data1)
data1=data1.iloc[:,top_cols]
data1=data1.as_matrix()
print(np.shape(data1))


#Load text data for each event and further tokenize the tweets which are considered as one sentence and one review will be 
#complete collection of tweets. 
data_train = pd.read_csv(server+"text.csv")
print (data_train.shape)
print(data_train.iloc[0])


reviews = []
labels = []
texts = []

##Data cleaning using BeautifulSoup and preprocessing library for tweets
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

##Data preprocessing to convert text data into 3 dimensional vector
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

##Convert credbility Labels to 1*4 vector for each event for ex: [1 0 0 0] for class 1 credible
labels = to_categorical(np.asarray(labels))
print(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

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
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SENT_LENGTH,
                            trainable=True,
                            mask_zero=True)
                            
#####============================================HAN Model==================================================================##

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



def create_model():

    # Words level attention model
	sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
	embedded_sequences = embedding_layer(sentence_input)
	l_lstm = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
	l_att = AttLayer(100)(l_lstm)
	sentEncoder = Model(sentence_input, l_att)

    # Tweet level attention model
	review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
	review_encoder = TimeDistributed(sentEncoder)(review_input)
	l_lstm_sent = Bidirectional(GRU(100, return_sequences=True))(review_encoder)
	l_att_sent = AttLayer(50)(l_lstm_sent)
	
    ##Augmenting socio linguistic feature vector with word embeddding vectors
	otherInp = Input(shape = (data1.shape[1], ))
	SocioLinguistic_sent=Dense(50)(otherInp)
	concatenatedFeatures = keras.layers.Concatenate(axis = -1)([l_att_sent, SocioLinguistic_sent])
	concatenatedFeatures_1 = Dense(50)(concatenatedFeatures)
	preds = Dense(4, activation='softmax')(concatenatedFeatures_1)
	model=Model(inputs=[review_input,otherInp], outputs=preds)
	model.summary()
	model.compile(loss='categorical_crossentropy',
		      optimizer='rmsprop',
		      metrics=['acc'])
	
	return model

#Random shuffle the data
np.random.seed(12)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
SocioLinguistic_data = data1[indices]

#Fitting the model with 5 cross validations
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.25, random_state=10)
cv_score=[]
for index, (train_indices, val_indices) in enumerate(sss.split(data,np.zeros(shape=(data.shape[0], 1)))):
    print ("Training on fold " + str(index+1) + "/5...")
    # Generate batches from indices
    x_train, x_val = data[train_indices], data[val_indices]
    y_train, y_val = labels[train_indices], labels[val_indices]
    SocioLinguistic_train,SocioLinguistic_val = SocioLinguistic_data[train_indices],SocioLinguistic_data[val_indices]
    print("SocioLinguistic data shape",SocioLinguistic_train.shape,SocioLinguistic_val.shape)

    print('Number of positive and negative reviews in training and validation set')
    print( y_train.sum(axis=0))
    print (y_val.sum(axis=0))
    # Clear model, and create it
    model = None
    model = create_model()
    


    print("model fitting - Hierachical attention network")
    model.fit([x_train,SocioLinguistic_train], y_train,validation_data=([x_val,SocioLinguistic_val],y_val),
              nb_epoch=3, batch_size=20, verbose=1)

    score=model.evaluate([x_val,SocioLinguistic_val],y_val)
    print(score)
    cv_score.append(score[1])
    result=[]
    for item in y_val:
        print(item)
        result.append(item)
    
    prediction=model.predict([x_val,SocioLinguistic_val])
    for item in prediction:
        result.append(item)
        print(item)
    with open(server+'HATT_all_10k_cv'+ str(index+1)+'.csv', 'w') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(result)

    model_json = model.to_json()
    with open(server+'HATT_all_5k_12_3_cv'+ str(index+1)+'.json', "w") as json_file:
         json_file.write(model_json)

print(cv_score.mean())

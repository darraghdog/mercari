# To do -- 1) Add count verctorizer to lgb with bigrams
# 2) replace the description with the tokenizer, see bm script
# 3) try Add() the embedding sequences
# 4) Check adding a linear model and blend
# 5) Give test a a larger batch size
# 6) Try cntk backend 
# mainly forking from notebook
# https://www.kaggle.com/johnfarrell/simple-rnn-with-keras-script
# http://jeffreyfossett.com/2014/04/25/tokenizing-raw-text-in-python.html
# encoding=utf8  
import sys  
import lightgbm as lgb
#reload(sys)  
#sys.setdefaultencoding('utf8')
import os, math, gc, time, random
import keras
from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
start_time = time.time()
import numpy as np
import re 
from numba import jit
from collections import Counter
from scipy.sparse import csr_matrix, hstack
import nltk
from nltk.tokenize import ToktokTokenizer
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import seaborn as sns
import tensorflow as tf
import keras
from keras.preprocessing.text import Tokenizer
def _get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)
_get_session()
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.layers import Input, Dropout, Dense, BatchNormalization, \
    Activation, concatenate, GRU, Embedding, Flatten, Bidirectional, \
    MaxPooling1D, Conv1D, Add, CuDNNLSTM, CuDNNGRU, Reshape
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping#, TensorBoard
from keras import backend as K
from keras import optimizers
from keras import initializers
from keras.utils import plot_model
import warnings
warnings.simplefilter(action='ignore')


os.chdir('/home/darragh/mercari/data')
#os.chdir('/Users/dhanley2/Documents/mercari/data')


train = pd.read_csv('../data/train.tsv', sep='\t', encoding='utf-8')
test = pd.read_csv('../data/test.tsv', sep='\t', encoding='utf-8')


train['target'] = np.log1p(train['price'])
print(train.shape)
print(test.shape)
print('[{}] Finished scaling test set...'.format(time.time() - start_time))

print("Remove bogus characters...")
@jit
def get_characters():
    characters = set()
    for sent in train.name.unique():
        for s in sent:
            characters.add(s)
    return characters
all_chars = sorted(list(get_characters()))
"".join(all_chars)


#text = u'This dog -- \uc758\uc774\uc8fd\ud55c\ud589\ud654\uf8ff\ufe0e -- \x96\xa1\xa2\xa3\xa4\xa5\xa6\xa7\xa9'
#print(text) # with emoji
# https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python
special_pattern = re.compile( 
    u"([\u0101-\ufffd])|"  
    u"([\x96-\xfc])" 
    "+", flags=re.UNICODE)
#print(special_pattern.sub(r'', text)) # no emoji


for col in ["item_description", "name", "brand_name"]:
    print("Clean special characters from " + col)
    train[col] = [(special_pattern.sub(r' ', sent)) if sent == sent else sent for sent in train[col].values]
    test[col] = [(special_pattern.sub(r' ', sent)) if sent == sent else sent for sent in test[col].values]

print('[{}] Finished remove bogus characters...'.format(time.time() - start_time))


#HANDLE MISSING VALUES
print("Handling missing values...")
def handle_missing(dataset):
    missing_string = "_missing_"
    dataset.category_name.fillna(value=missing_string, inplace=True)
    dataset.brand_name.fillna(value=missing_string, inplace=True)
    dataset.item_description.fillna(value=missing_string, inplace=True)
    return (dataset)

train = handle_missing(train)
test = handle_missing(test)
print(train.shape)
print(test.shape)
print('[{}] Finished handling missing data...'.format(time.time() - start_time))

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
print("Handling categorical variables...")
le = LabelEncoder()

le.fit(np.hstack([train.category_name, test.category_name]))
train['category'] = le.transform(train.category_name)
test['category'] = le.transform(test.category_name)


hi_brand_cts = train['brand_name'].value_counts()
hi_brand_cts = hi_brand_cts[hi_brand_cts>5].index.values
train.brand_name[~train.brand_name.isin(hi_brand_cts)] = '_lo_count_'
test.brand_name[~test.brand_name.isin(hi_brand_cts)] = '_lo_count_'
le.fit(np.hstack([train.brand_name, test.brand_name]))
train['brand'] = le.transform(train.brand_name)
test['brand'] = le.transform(test.brand_name)
del le#, train['brand_name'], test['brand_name']

# Replace the category slash
test["category_name_split"] = test["category_name"].str.replace(' ', '_')
train["category_name_split"] = train["category_name"].str.replace(' ', '_')
test["category_name_split"] = test["category_name_split"].str.replace('/', ' ')
train["category_name_split"] = train["category_name_split"].str.replace('/', ' ')

print('[{}] Start tokenising...'.format(time.time() - start_time))

toktok = ToktokTokenizer()
token_map = {}
def nltkToken(sent):
    if sent in token_map:
        return token_map[sent]
    else:
        token_map[sent] = toktok.tokenize(sent)
        return token_map[sent]

train['name_token'] = [nltkToken(sent) for sent in train['name'].str.lower().str.replace('/', ' ').tolist()]
test ['name_token'] = [nltkToken(sent) for sent in test['name'].str.lower().str.replace('/', ' ').tolist()]
print('[{}] Part 1...'.format(time.time() - start_time))
train['cat_token'] = [nltkToken(sent) for sent in train["category_name"].str.lower().str.replace('/', ' ').tolist()]
test ['cat_token'] = [nltkToken(sent) for sent in test["category_name"].str.lower().str.replace('/', ' ').tolist()]
print('[{}] Part 2...'.format(time.time() - start_time))
#train['descr_token'] = [nltkToken(sent[:150]) for sent in train['item_description'].str.lower().str.replace('/', ' ').tolist()]
#test ['descr_token'] = [nltkToken(sent[:150]) for sent in test['item_description'].str.lower().str.replace('/', ' ').tolist()]
#print('[{}] Part 3...'.format(time.time() - start_time))
train['brand_token'] = [nltkToken(sent) for sent in train['brand_name'].str.lower().str.replace('/', ' ').tolist()]
test ['brand_token'] = [nltkToken(sent) for sent in test ['brand_name'].str.lower().str.replace('/', ' ').tolist()]
print('[{}] Finished Tokenizing text...'.format(time.time() - start_time))

@jit
def list_flatten(var):
    list_ = []
    for sent_ in var:
        list_ += sent_
    return Counter(list_)

wordlist = []
for col in ['name_token', 'cat_token', 'descr_token', 'brand_name']:
    for dt in train, test:
        flat_counter = list_flatten(dt[[col]].values[:,0])
        wordlist += [k for (k, v) in flat_counter.items() if v>2]
        wordlist = list(set(wordlist))
wordlist = set(wordlist)
#
embeddings_matrix = []
embedding_map = {}
f = open('../feat/wiki.en.vec')
counter = 0
for line in f:
    values = line.split()
    word = values[0]
    if word not in wordlist:
        continue
    #coefs = np.asarray(values[1:], dtype='float32')
    embedding_map[word] = counter
    embeddings_matrix.append(values[1:])
    counter += 1
    if (counter % 10000 == 0) and (counter != 0):
        print('Found %s word vectors.' % counter)
f.close()
print('Found %s word vectors.' % counter)

embeddings_matrix = np.array(embeddings_matrix, dtype='float32')

print('[{}] Embeddings loaded...'.format(time.time() - start_time))

dtrain, dvalid = train_test_split(train, random_state=233, train_size=0.90)


# Get the dot product
def posn_to_sparse(dt, embedding_map):
    sprow = []
    spcol = []
    spdata = []
    for c, (nm, cat, dscr) in enumerate(zip(dt['name_token'].values, 
                            dt['cat_token'].values, 
                            dt['descr_token'].values)):
        sent = nm + cat + dscr
        ids = [embedding_map[s] for s in sent if s in embedding_map]
        n_ = len(ids)
        sprow += [c]*n_
        spcol += ids
        spdata += [1]*n_    
    shape_ = (dt.shape[0], len(embedding_map.keys())) 
    dt_ids = csr_matrix((spdata, (sprow, spcol)), shape=shape_)
    return dt_ids

dtrain_ids = posn_to_sparse(dtrain, embedding_map)
dvalid_ids = posn_to_sparse(dvalid, embedding_map)
test_ids  = posn_to_sparse(test, embedding_map)

## Make the embeddig matrix sparse
#embeddings_matrix = csr_matrix(embeddings_matrix)
# Get the dense layer input of the text
densetrn = dtrain_ids.dot(embeddings_matrix)#.todense()
denseval = dvalid_ids.dot(embeddings_matrix)#.todense()
densetst = test_ids.dot(embeddings_matrix)#.todense()
densetst
mean_, sd_ = densetrn.mean(), densetrn.std()
densetrn -= mean_
denseval -= mean_
densetst -= mean_
densetrn /= sd_
denseval /= sd_
densetst /= sd_

print(dtrain.shape)
print(densetrn.shape)

def get_keras_data(denseset, dataset):
    X = {
        'sent_emb': np.array(denseset)
        ,'item_condition': np.array(dataset.item_condition_id)
        ,'num_vars': np.array(dataset[["shipping"]])
        , 'cat' : np.array(dataset.category)
        , 'brand' : np.array(dataset.brand)
    }
    return X   

def reset_data(dt, bsize):
    max_step = dt.shape[0]
    n_batches = int(np.ceil(max_step*1. / float(bsize)))
    batch_steps = np.array(random.sample(range(n_batches), n_batches))
    return max_step, batch_steps

def trn_generator(dn, dt, y, bsize):
    while True:
        max_step, batch_steps = reset_data(dt, bsize)
        for batch in batch_steps:
            from_ = batch*bsize
            to_   = min((batch+1)*bsize, max_step)
            Xbatch = dt.iloc[from_:to_]
            Xdbatch = dn[from_:to_]
            Xbatch = get_keras_data(Xdbatch, Xbatch)
            ybatch = dt.target.iloc[from_:to_]
            yield Xbatch, ybatch

def val_generator(dn, dt, y, bsize):
    while 1:
        max_step, batch_steps = reset_data(dt, bsize)
        for batch in batch_steps:
            from_ = batch*bsize
            to_   = min((batch+1)*bsize, max_step)
            Xbatch = dt.iloc[from_:to_]
            Xdbatch = dn[from_:to_]
            Xbatch = get_keras_data(Xdbatch, Xbatch)
            ybatch = dt.target.iloc[from_:to_]
            yield Xbatch, ybatch

print('[{}] Dense data creates...'.format(time.time() - start_time))
train.head()

MAX_CAT   = dtrain.category.values.max()
MAX_BRAND = dtrain.brand.values.max()

def dense_model():
    dr = 0.0

    ##Inputs=
    item_condition = Input(shape=[1], name="item_condition")
    cat = Input(shape=[1], name="cat")
    brand = Input(shape=[1], name="brand")
    num_vars = Input(shape=[1], name="num_vars")
    sent_emb = Input(shape=[300], name="sent_emb")
    
    #Embeddings layers
    emb_item_condition      = Embedding(6, 5)(item_condition)
    cat_item_condition      = Embedding(MAX_CAT, 20)(cat)
    brand_item_condition      = Embedding(MAX_BRAND, 20)(brand)
    
    #main layer
    main_l = concatenate([
        sent_emb
        , Flatten() (emb_item_condition)
        , Flatten() (cat_item_condition)
        , Flatten() (brand_item_condition)
        , num_vars
    ])
    
    main_l = Dropout(dr)(Dense(1024,activation='relu') (main_l))
    main_l = Dropout(dr)(Dense(1024,activation='relu') (main_l))
    
    main_l = Dropout(dr)(Dense(512,activation='relu') (main_l))
    main_l = Dropout(dr)(Dense(512,activation='relu') (main_l))
    
    main_l = Dropout(dr)(Dense(256,activation='relu') (main_l))
    main_l = Dropout(dr)(Dense(256,activation='relu') (main_l))
    
    main_l = Dropout(dr)(Dense(128,activation='relu') (main_l))
    main_l = Dropout(dr)(Dense(128,activation='relu') (main_l))
    #output
    output = Dense(1,activation="linear") (main_l)
    
    #model
    model = Model([sent_emb, item_condition, num_vars, cat, brand], output)
    optimizer = optimizers.Adam()
    model.compile(loss='mse', 
                  optimizer=optimizer)
    return model

print('[{}] Finished DEFINING MODEL...'.format(time.time() - start_time))

epochs = 20
batchSize = 512 * 4
steps = (dtrain.shape[0]/batchSize+1)*epochs
lr_init, lr_fin = 0.001, 0.0001
lr_decay  = (lr_init - lr_fin)/steps
model = dense_model()
K.set_value(model.optimizer.lr, lr_init)
K.set_value(model.optimizer.decay, lr_decay)

model.fit_generator(
                    trn_generator(densetrn, dtrain, dtrain.target, batchSize)
                    , epochs=epochs
                    , max_queue_size=1
                    , steps_per_epoch = int(np.ceil(dtrain.shape[0]*1./batchSize))
                    , validation_data = val_generator(denseval, dvalid, dvalid.target, batchSize)
                    , validation_steps = int(np.ceil(dvalid.shape[0]*1./batchSize))
                    , verbose=2
                    )
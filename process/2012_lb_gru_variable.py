# coding: utf-8

# mainly forking from notebook
# https://www.kaggle.com/johnfarrell/simple-rnn-with-keras-script

import os, math, gc, time, random
start_time = time.time()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
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

os.chdir('/home/darragh/mercari/data')
#os.chdir('/Users/dhanley2/Documents/mercari/data')


train = pd.read_csv('../data/train.tsv', sep='\t')
test = pd.read_csv('../data/test.tsv', sep='\t')

train['target'] = np.log1p(train['price'])
print(train.shape)
print(test.shape)
print('[{}] Finished scaling test set...'.format(time.time() - start_time))


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

pd.set_option('display.height', 500)
pd.set_option('display.max_rows', 500)

#PROCESS CATEGORICAL DATA
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
print("Handling categorical variables...")
le = LabelEncoder()

le.fit(np.hstack([train.category_name, test.category_name]))
train['category'] = le.transform(train.category_name)
test['category'] = le.transform(test.category_name)


hi_brand_cts = train['brand_name'].value_counts()
hi_brand_cts = brand_cts[brand_cts>5].index.values
train.brand_name[~train.brand_name.isin(hi_brand_cts)] = '_lo_count_'
test.brand_name[~test.brand_name.isin(hi_brand_cts)] = '_lo_count_'
le.fit(np.hstack([train.brand_name, test.brand_name]))
train['brand'] = le.transform(train.brand_name)
test['brand'] = le.transform(test.brand_name)
del le, train['brand_name'], test['brand_name']

# Replace the category slash
test["category_name_split"] = test["category_name"].str.replace(' ', '_')
train["category_name_split"] = train["category_name"].str.replace(' ', '_')
test["category_name_split"] = test["category_name_split"].str.replace('/', ' ')
train["category_name_split"] = train["category_name_split"].str.replace('/', ' ')
train.head()
print('[{}] Finished PROCESSING CATEGORICAL DATA...'.format(time.time() - start_time))


#PROCESS TEXT: RAW
print("Text to seq process...")
print("   Fitting tokenizer...")
tok_raw_cat = Tokenizer()
tok_raw_nam = Tokenizer()
tok_raw_dsc = Tokenizer()
tok_raw_cat.fit_on_texts(train.category_name_split.str.lower().unique())
tok_raw_nam.fit_on_texts(train.name.str.lower().unique())
tok_raw_dsc.fit_on_texts(train.item_description.str.lower().unique())

print("   Transforming text to seq...")
train["seq_category_name_split"] = tok_raw_cat.texts_to_sequences(train.category_name_split.str.lower())
test["seq_category_name_split"] =  tok_raw_cat.texts_to_sequences(test.category_name_split.str.lower())
train["seq_item_description"] =    tok_raw_dsc.texts_to_sequences(train.item_description.str.lower())
test["seq_item_description"] =     tok_raw_dsc.texts_to_sequences(test.item_description.str.lower())
train["seq_name"] =                tok_raw_nam.texts_to_sequences(train.name.str.lower())
test["seq_name"] =                 tok_raw_nam.texts_to_sequences(test.name.str.lower())
print('[{}] Finished PROCESSING TEXT DATA...'.format(time.time() - start_time))


#EXTRACT DEVELOPTMENT TEST
dtrain, dvalid = train_test_split(train, random_state=233, train_size=0.90)
print(dtrain.shape)
print(dvalid.shape)


#EMBEDDINGS MAX VALUE
MAX_CAT = len(tok_raw_cat.word_counts)+1
MAX_NAM = len(tok_raw_nam.word_counts)+1
MAX_DSC = len(tok_raw_dsc.word_counts)+1
MAX_CATEGORY = np.max([train.category.max(), test.category.max()])+1
MAX_BRAND = np.max([train.brand.max(), test.brand.max()])+1
MAX_CONDITION = np.max([train.item_condition_id.max(), 
                        test.item_condition_id.max()])+1


print('[{}] Finished EMBEDDINGS MAX VALUE...'.format(time.time() - start_time)) 

# Sort the data on the length of the descriptions which will make it quicker
# for the variable length GRU
def len_argsort(seq):
	return sorted(range(len(seq)), key=lambda x: len(seq[x]))

def get_keras_data(dataset):
    X = {
        'name': pad_sequences(dataset.seq_name, 
                              maxlen=max([len(l) for l in dataset.seq_name]))
        ,'item_desc': pad_sequences(dataset.seq_item_description, 
                              maxlen=max([len(l) for l in dataset.seq_item_description]))
        ,'brand': np.array(dataset.brand)
        ,'category': np.array(dataset.category)
        ,'category_name_split': pad_sequences(dataset.seq_category_name_split, 
                              maxlen=max([len(l) for l in dataset.seq_category_name_split]))
        ,'item_condition': np.array(dataset.item_condition_id)
        ,'num_vars': np.array(dataset[["shipping"]])
    }
    return X   

def eval_model(y_true, val_preds):
    val_preds = np.expm1(val_preds)
    y_pred = val_preds[:, 0]
    v_rmsle = rmsle(y_true, y_pred)
    print("RMSLE error on dev test: "+str(v_rmsle))
    return v_rmsle

def reset_data(dt, bsize):
    max_step = dt.shape[0]
    n_batches = int(np.ceil(max_step*1. / float(bsize)))
    batch_steps = np.array(random.sample(range(n_batches), n_batches))
    sorted_ix = np.array(len_argsort(dt["seq_item_description"].tolist()))
    dt.reset_index(drop=True, inplace = True)  
    return max_step, batch_steps, sorted_ix, dt

def trn_generator(dt, y, bsize):
    while True:
        max_step, batch_steps, sorted_ix, dt = reset_data(dt, bsize)
        for batch in batch_steps:
            from_ = batch*bsize
            to_   = min((batch+1)*bsize, max_step)
            ix_   = sorted_ix[from_:to_]
            Xbatch = dt.iloc[ix_]
            Xbatch = get_keras_data(Xbatch)
            ybatch = dt.target.iloc[ix_]
            yield Xbatch, ybatch

def val_generator(dt, y, bsize):
    while 1:
        max_step, batch_steps, sorted_ix, dt = reset_data(dt, bsize)
        for batch in batch_steps:
            from_ = batch*bsize
            to_   = min((batch+1)*bsize, max_step)
            ix_   = sorted_ix[from_:to_]
            Xbatch = dt.iloc[ix_]
            Xbatch = get_keras_data(Xbatch)
            ybatch = dt.target.iloc[ix_]
            yield Xbatch, ybatch

def tst_generator(dt, bsize):
    while 1:
        for batch in range(dt.shape[0]/bsize+1):
            from_ = batch*bsize
            to_   = min((batch+1)*bsize, dt.shape[0])
            Xbatch = dt.iloc[from_:to_]
            Xbatch = get_keras_data(Xbatch)
            yield Xbatch

#KERAS MODEL DEFINITION
def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 \
              for i, pred in enumerate(y_pred)]
    return (sum(to_sum) * (1.0/len(y))) ** 0.5

dr = 0.1

from keras.layers import GlobalMaxPooling1D

def get_model():

    ##Inputs
    name = Input(shape=[None], name="name")
    item_desc = Input(shape=[None], name="item_desc")
    category_name_split = Input(shape=[None], name="category_name_split")
    brand = Input(shape=[1], name="brand")
    item_condition = Input(shape=[1], name="item_condition")
    num_vars = Input(shape=[1], name="num_vars")
    
    #Embeddings layers
    emb_size = 60
    emb_name                = Embedding(MAX_NAM, emb_size//2)(name) 
    emb_item_desc           = Embedding(MAX_DSC, emb_size)(item_desc) 
    emb_category_name_split = Embedding(MAX_CAT, emb_size//3)(category_name_split) 
    emb_brand               = Embedding(MAX_BRAND, 8)(brand)
    emb_item_condition      = Embedding(MAX_CONDITION, 5)(item_condition)
    
    rnn_layer1 = GRU(16, recurrent_dropout=0.0) (emb_item_desc)
    rnn_layer2 = GRU(8, recurrent_dropout=0.0) (emb_category_name_split)
    rnn_layer3 = GRU(8, recurrent_dropout=0.0) (emb_name)
    
    #main layer
    main_l = concatenate([
        Flatten() (emb_brand)
        , Flatten() (emb_item_condition)
        , rnn_layer1
        , rnn_layer2
        , rnn_layer3
        , num_vars
    ])
    main_l = Dropout(dr*2)(Dense(128,activation='relu') (main_l))
    main_l = Dropout(dr)(Dense(64,activation='relu') (main_l))
    
    #output
    output = Dense(1,activation="linear") (main_l)
    
    #model
    model = Model([name, item_desc, brand
                   , category_name_split #,category
                   , item_condition, num_vars], output)
    optimizer = optimizers.Adam()
    model.compile(loss='mse', 
                  optimizer=optimizer)
    return model

print('[{}] Finished DEFINING MODEL...'.format(time.time() - start_time))


#dt, y, batchSize = dvalid, dvalid.target, batchSize_test
epochs = 3
batchSize = 512 * 3
steps = (dtrain.shape[0]/batchSize+1)*epochs
lr_init, lr_fin = 0.015, 0.007
lr_decay  = (lr_init - lr_fin)/steps
model = get_model()
K.set_value(model.optimizer.lr, lr_init)
K.set_value(model.optimizer.decay, lr_decay)


history = model.fit_generator(
                    trn_generator(dtrain, dtrain.target, batchSize)
                    , epochs=epochs
                    , max_queue_size=1
                    , steps_per_epoch = dtrain.shape[0]/batchSize+1
                    , validation_data = val_generator(dvalid, dvalid.target, batchSize)
                    , validation_steps = dvalid.shape[0]/batchSize+1
                    , verbose=1
                    )
val_sorted_ix = np.array(len_argsort(dvalid["seq_item_description"].tolist()))
y_pred = model.predict_generator(tst_generator(dvalid.iloc[val_sorted_ix], batchSize)
                    , steps = dvalid.shape[0]/batchSize+1
                    , max_queue_size=1 
                    , verbose=1)
print "Epoch %s rmsle %s"%(e, eval_model(dvalid.price.values[val_sorted_ix], y_pred))


'''
y_pred = model.predict_generator(tst_generator(test, batchSize), 
                         steps = test.shape[0]/batchSize+1, 
                         verbose=1)
    

    np.histogram(y_pred)
    np.histogram( dvalid.target[val_sorted_ix])


print('[{}] Finished FITTING MODEL...'.format(time.time() - start_time))

#CREATE PREDICTIONS
preds = model.predict(X_test, batch_size=BATCH_SIZE)
preds = np.expm1(preds)
print('[{}] Finished predicting test set...'.format(time.time() - start_time))
submission = test[["test_id"]][:test_len]
submission["price"] = preds[:test_len]
submission.to_csv("./myNN"+log_subdir+"_{:.6}.csv".format(v_rmsle), index=False)
print('[{}] Finished submission...'.format(time.time() - start_time))
'''
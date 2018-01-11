# coding: utf-8

# mainly forking from notebook
# https://www.kaggle.com/johnfarrell/simple-rnn-with-keras-script

import os, math, gc, time
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
    Activation, concatenate, GRU, Embedding, Flatten, Bidirectional, MaxPooling1D, Conv1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping#, TensorBoard
from keras import backend as K
from keras import optimizers
from keras import initializers

os.chdir('/home/darragh/mercari/data')


train = pd.read_csv('../data/train.tsv', sep='\t')
test = pd.read_csv('../data/test.tsv', sep='\t')

train['target'] = np.log1p(train['price'])
print(train.shape)
print('5 folds scaling the test_df')
print(test.shape)
test_len = test.shape[0]
def simulate_test(test):
    if test.shape[0] < 800000:
        indices = np.random.choice(test.index.values, 2800000)
        test_ = pd.concat([test, test.iloc[indices]], axis=0)
        return test_.copy()
    else:
        return test
test = simulate_test(test)
print('new shape ', test.shape)
print('[{}] Finished scaling test set...'.format(time.time() - start_time))


#HANDLE MISSING VALUES
print("Handling missing values...")
def handle_missing(dataset):
    dataset.category_name.fillna(value="missing", inplace=True)
    dataset.brand_name.fillna(value="missing", inplace=True)
    dataset.item_description.fillna(value="missing", inplace=True)
    return (dataset)

train = handle_missing(train)
test = handle_missing(test)
print(train.shape)
print(test.shape)

print('[{}] Finished handling missing data...'.format(time.time() - start_time))


#PROCESS CATEGORICAL DATA
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
print("Handling categorical variables...")
le = LabelEncoder()

le.fit(np.hstack([train.category_name, test.category_name]))
train['category'] = le.transform(train.category_name)
test['category'] = le.transform(test.category_name)

le.fit(np.hstack([train.brand_name, test.brand_name]))
train['brand'] = le.transform(train.brand_name)
test['brand'] = le.transform(test.brand_name)
del le, train['brand_name'], test['brand_name']

# Stplit the category to three columns
train[['cat1','cat2', 'cat3']] = pd.DataFrame(train["category_name"].str.split('/',2).tolist())
test[['cat1','cat2', 'cat3']] = pd.DataFrame(test["category_name"].str.split('/',2).tolist())

# Replace the category slash
test["category_name_split"] = test["category_name"].str.replace('/', ' ')
train["category_name_split"] = train["category_name"].str.replace('/', ' ')
train.head()

print('[{}] Finished PROCESSING CATEGORICAL DATA...'.format(time.time() - start_time))
train.head(3)

#PROCESS TEXT: RAW
print("Text to seq process...")
print("   Fitting tokenizer...")
raw_text = np.hstack([train.category_name.str.lower().unique(), 
                      train.category_name_split.str.lower().unique(), 
                      train.item_description.str.lower().unique(), 
                      train.name.str.lower().unique()])

tok_raw = Tokenizer()
tok_raw.fit_on_texts(raw_text)
print("   Transforming text to seq...")
#train["seq_category_name"] = tok_raw.texts_to_sequences(train.category_name.str.lower())
#test["seq_category_name"] = tok_raw.texts_to_sequences(test.category_name.str.lower())
train["seq_category_name_split"] = tok_raw.texts_to_sequences(train.category_name_split.str.lower())
test["seq_category_name_split"] =  tok_raw.texts_to_sequences(test.category_name_split.str.lower())
train["seq_item_description"] =    tok_raw.texts_to_sequences(train.item_description.str.lower())
test["seq_item_description"] =     tok_raw.texts_to_sequences(test.item_description.str.lower())
train["seq_name"] =                tok_raw.texts_to_sequences(train.name.str.lower())
test["seq_name"] =                 tok_raw.texts_to_sequences(test.name.str.lower())
for col in ['cat1','cat2', 'cat3']:
    le = LabelEncoder()
    le.fit(train[col].unique().tolist() + test[col].unique().tolist())    
    train[col] = le.transform(train[col].tolist())
    test[col] = le.transform(test[col].tolist())



print('[{}] Finished PROCESSING TEXT DATA...'.format(time.time() - start_time))

#EXTRACT DEVELOPTMENT TEST
dtrain, dvalid = train_test_split(train, random_state=233, train_size=0.90)
print(dtrain.shape)
print(dvalid.shape)


#EMBEDDINGS MAX VALUE
#Base on the histograms, we select the next lengths
MAX_NAME_SEQ = 20 #17
MAX_ITEM_DESC_SEQ = 80 #269
MAX_CATEGORY_NAME_SEQ = 20 #8
MAX_CATEGORY_NAME_SPLIT_SEQ = 20
MAX_TEXT_CATEGORY_NAME = np.max([
                   np.max(train.seq_category_name_split.max())
                   , np.max(test.seq_category_name_split.max())])+2
MAX_TEXT = np.max([np.max(train.seq_name.max())
                   , np.max(test.seq_name.max())
                   #, np.max(train.seq_category_name.max())
                   #, np.max(test.seq_category_name.max())
                   , np.max(train.seq_category_name_split.max())
                   , np.max(test.seq_category_name_split.max())
                   , np.max(train.seq_item_description.max())
                   , np.max(test.seq_item_description.max())])+2
MAX_CATEGORY = np.max([train.category.max(), test.category.max()])+1
MAX_BRAND = np.max([train.brand.max(), test.brand.max()])+1
MAX_CONDITION = np.max([train.item_condition_id.max(), 
                        test.item_condition_id.max()])+1

MAX_CAT1 = np.max([train.cat1.max(), test.cat1.max()])+1
MAX_CAT2 = np.max([train.cat2.max(), test.cat2.max()])+1
MAX_CAT3 = np.max([train.cat3.max(), test.cat3.max()])+1
    

print('[{}] Finished EMBEDDINGS MAX VALUE...'.format(time.time() - start_time))

#KERAS DATA DEFINITION
def get_keras_data(dataset):
    X = {
        'name': pad_sequences(dataset.seq_name, maxlen=MAX_NAME_SEQ)
        ,'item_desc': pad_sequences(dataset.seq_item_description
                                    , maxlen=MAX_ITEM_DESC_SEQ)
        ,'brand': np.array(dataset.brand)
        ,'category': np.array(dataset.category)
        ,'cat1': np.array(dataset.cat1)
        ,'cat2': np.array(dataset.cat2)
        ,'cat3': np.array(dataset.cat3)
        #,'category_name': pad_sequences(dataset.seq_category_name
        #                                , maxlen=MAX_CATEGORY_NAME_SEQ)
        ,'category_name_split': pad_sequences(dataset.seq_category_name_split
                                        , maxlen=MAX_CATEGORY_NAME_SPLIT_SEQ)
        ,'item_condition': np.array(dataset.item_condition_id)
        ,'num_vars': np.array(dataset[["shipping"]])
    }
    return X

X_train = get_keras_data(dtrain)
X_valid = get_keras_data(dvalid)
X_test = get_keras_data(test)

print('[{}] Finished DATA PREPARARTION...'.format(time.time() - start_time))

#KERAS MODEL DEFINITION
def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 \
              for i, pred in enumerate(y_pred)]
    return (sum(to_sum) * (1.0/len(y))) ** 0.5

dr = 0.2

def get_model():
    #Inputs
    name = Input(shape=[X_train["name"].shape[1]], name="name")
    item_desc = Input(shape=[X_train["item_desc"].shape[1]], name="item_desc")
    brand = Input(shape=[1], name="brand")
    #category = Input(shape=[1], name="category")
    category_name_split = Input(shape=[X_train["category_name_split"].shape[1]], 
                          name="category_name_split")
    item_condition = Input(shape=[1], name="item_condition")
    num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")
    
    #Embeddings layers
    emb_size = 60
    
    emb_name = Embedding(MAX_TEXT, emb_size)(name) # , mask_zero=True
    emb_item_desc = Embedding(MAX_TEXT, emb_size)(item_desc) # , mask_zero=True#
    emb_category_name_split = Embedding(MAX_TEXT, emb_size//3)(category_name_split) # , mask_zero=True
    emb_brand = Embedding(MAX_BRAND, 8)(brand)
    emb_item_condition = Embedding(MAX_CONDITION, 5)(item_condition)
    
    conv1   = Conv1D(16, 3, activation='relu', padding = "same") (emb_item_desc)
    conv1   = MaxPooling1D(2)(conv1)
    conv1   = Conv1D(32, 3, activation='relu', padding = "same") (conv1)
    conv1   = MaxPooling1D(2)(conv1)
    conv1   = Conv1D(32, 3, activation='relu', padding = "same") (conv1)
    conv1   = MaxPooling1D(2)(conv1)
    
    conv2   = Conv1D(16, 3, activation='relu', padding = "same") (emb_category_name_split)
    conv2   = MaxPooling1D(2)(conv2)    
    conv2   = Conv1D(32, 3, activation='relu', padding = "same") (conv2)
    #conv2   = MaxPooling1D(2)(conv2)
    
    conv3   = Conv1D(16, 3, activation='relu', padding = "same") (emb_name)
    conv3   = MaxPooling1D(2)(conv3)
    conv3   = Conv1D(32, 3, activation='relu', padding = "same") (conv3)
    #conv3   = MaxPooling1D(2)(conv3)
    rnn_layer3 = GRU(32, recurrent_dropout=0.0) (concatenate([conv3, conv2, conv1]))
        
    #main layer
    main_l = concatenate([
        Flatten() (emb_brand)
        #, Flatten() (emb_category)
        , Flatten() (emb_item_condition)
        #, rnn_layer1
        #, rnn_layer2
        , rnn_layer3
        , num_vars
    ])
    main_l = Dropout(dr)(Dense(512,activation='relu') (main_l))
    main_l = Dropout(dr)(Dense(64,activation='relu') (main_l))
    
    #output
    output = Dense(1,activation="linear") (main_l)
    
    #model
    model = Model([name, item_desc, brand#, cat1, cat2, cat3
                   , category_name_split# category_name, category, 
                   , item_condition, num_vars], output)
    #optimizer = optimizers.RMSprop()
    optimizer = optimizers.Adam()
    model.compile(loss='mse', 
                  optimizer=optimizer)
    return model

def eval_model(model, batch_size, epoch):
    val_preds = model.predict(X_valid, batch_size=batch_size)
    val_preds = np.expm1(val_preds)
    
    y_true = np.array(dvalid.price.values)
    y_pred = val_preds[:, 0]
    v_rmsle = rmsle(y_true, y_pred)
    print("Epoch ",str(epoch)," RMSLE error on dev test: "+str(v_rmsle))
    return v_rmsle

print('[{}] Finished DEFINEING MODEL...'.format(time.time() - start_time))

gc.collect()
epochs = 3
BATCH_SIZE = 512 * 4
steps = int(len(X_train['name'])/BATCH_SIZE) * epochs
lr_init = 0.01

model = get_model()
K.set_value(model.optimizer.lr, lr_init)

for i in range(epochs):
    history = model.fit(X_train, dtrain.target
                        , epochs=1#epochs
                        , batch_size=BATCH_SIZE
                        #, validation_split=0.1
                        #, callbacks=[TensorBoard('./logs/'+log_subdir)]
                        , validation_data = (X_valid, dvalid.target)
                        , verbose=1
                        )
    v_rmsle = eval_model(model, batch_size=BATCH_SIZE, epoch = i)
    # Baseline @20epochs -- val_loss: 0.2190; RMSLE error on dev test: 0.4679
    # Baseline @8epochs,dr0.3 -- val_loss: 0.2057; RMSLE error on dev test: 0.4535
    # Baseline @8epochs,dr0.3,3xemb -- val_loss: 0.1942; RMSLE error on dev test: 0.4406
    # Baseline @8epochs,dr0.4,3xemb, properseq -- val_loss: 0.1898; RMSLE error on dev test: 0.4356
    # Baseline @8epochs,dr0.2,3xemb, properseq -- val_loss: 0.1896; RMSLE error on dev test: 0.4353
    #EVLUEATE THE MODEL ON DEV TEST

print('[{}] Finished FITTING MODEL...'.format(time.time() - start_time))

#CREATE PREDICTIONS
preds = model.predict(X_test, batch_size=BATCH_SIZE)
preds = np.expm1(preds)
print('[{}] Finished predicting test set...'.format(time.time() - start_time))
submission = test[["test_id"]][:test_len]
submission["price"] = preds[:test_len]
submission.to_csv("./myNN"+log_subdir+"_{:.6}.csv".format(v_rmsle), index=False)
print('[{}] Finished submission...'.format(time.time() - start_time))
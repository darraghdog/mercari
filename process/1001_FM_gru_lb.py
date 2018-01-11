# coding: utf-8

# mainly forking from notebook
# https://www.kaggle.com/johnfarrell/simple-rnn-with-keras-script
# http://jeffreyfossett.com/2014/04/25/tokenizing-raw-text-in-python.html
# encoding=utf8  
import lightgbm as lgb
import sys  
#reload(sys)  
#sys.setdefaultencoding('utf8')
import os, math, gc, time, random
start_time = time.time()
import numpy as np
from numba import jit
from collections import Counter
from scipy.sparse import csr_matrix, hstack
import nltk, re
from nltk.tokenize import ToktokTokenizer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
import multiprocessing as mp
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
    MaxPooling1D, Conv1D, Add, Reshape, Lambda
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping#, TensorBoard
from keras import backend as K
from keras import optimizers
from keras import initializers
from keras.utils import plot_model
import warnings
warnings.simplefilter(action='ignore')
# Based on Bojan -> https://www.kaggle.com/tunguz/more-effective-ridge-lgbm-script-lb-0-44944
# and Nishant -> https://www.kaggle.com/nishkgp/more-improved-ridge-2-lgbm

import gc
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import lightgbm as lgb

import sys

#Add https://www.kaggle.com/anttip/wordbatch to your kernel Data Sources, 
#until Kaggle admins fix the wordbatch pip package installation
sys.path.insert(0, '../input/wordbatch/wordbatch/')
import wordbatch

from wordbatch.extractors import WordBag, WordHash
from wordbatch.models import FTRL, FM_FTRL

from nltk.corpus import stopwords
import re
start_time = time.time()
from time import gmtime, strftime


os.chdir('/home/darragh/mercari/data')
#os.chdir('/Users/dhanley2/Documents/mercari/data')


train = pd.read_csv('../data/train.tsv', sep='\t', encoding='utf-8')
test = pd.read_csv('../data/test.tsv', sep='\t', encoding='utf-8')
glove_file = '../feat/glove.6B.50d.txt'
threads = 2

##EXTRACT DEVELOPTMENT TEST
#dtrain, dvalid = train_test_split(train, random_state=233, train_size=0.90)


NUM_BRANDS = 4500
NUM_CATEGORIES = 1200

develop = False
develop= True

def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y0), 2)))


def split_cat(text):
    try:
        return text.split("/")
    except:
        return ("No Label", "No Label", "No Label")


def handle_missing_inplace(dataset):
    dataset['general_cat'].fillna(value='missing', inplace=True)
    dataset['subcat_1'].fillna(value='missing', inplace=True)
    dataset['subcat_2'].fillna(value='missing', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='missing', inplace=True)


def cutting(dataset):
    pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'
    pop_category1 = dataset['general_cat'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category2 = dataset['subcat_1'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category3 = dataset['subcat_2'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    dataset.loc[~dataset['general_cat'].isin(pop_category1), 'general_cat'] = 'missing'
    dataset.loc[~dataset['subcat_1'].isin(pop_category2), 'subcat_1'] = 'missing'
    dataset.loc[~dataset['subcat_2'].isin(pop_category3), 'subcat_2'] = 'missing'


def to_categorical(dataset):
    dataset['general_cat'] = dataset['general_cat'].astype('category')
    dataset['subcat_1'] = dataset['subcat_1'].astype('category')
    dataset['subcat_2'] = dataset['subcat_2'].astype('category')
    dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')


# Define helpers for text normalization
stopwords = {x: 1 for x in stopwords.words('english')}
non_alphanums = re.compile(u'[^A-Za-z0-9]+')


def normalize_text(text):
    return u" ".join(
        [x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] \
         if len(x) > 1 and x not in stopwords])

print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))

print('[{}] Finished to load data'.format(time.time() - start_time))
print('Train shape: ', train.shape)
print('Test shape: ', test.shape)
nrow_test = train.shape[0]  # -dftt.shape[0]
dftt = train[(train.price < 1.0)]
train = train.drop(train[(train.price < 1.0)].index)
del dftt['price']
nrow_train = train.shape[0]
# print(nrow_train, nrow_test)
y = np.log1p(train["price"])
merge = pd.concat([train, dftt, test])
submission = test[['test_id']]

del train
del test
gc.collect()

merge['general_cat'], merge['subcat_1'], merge['subcat_2'] = \
    zip(*merge['category_name'].apply(lambda x: split_cat(x)))
merge.drop('category_name', axis=1, inplace=True)
print('[{}] Split categories completed.'.format(time.time() - start_time))

handle_missing_inplace(merge)
print('[{}] Handle missing completed.'.format(time.time() - start_time))

cutting(merge)
print('[{}] Cut completed.'.format(time.time() - start_time))

to_categorical(merge)
print('[{}] Convert categorical completed'.format(time.time() - start_time))

wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.5, 1.0],
                                                              "hash_size": 2 ** 29, "norm": None, "tf": 'binary',
                                                              "idf": None,
                                                              }), procs=8)
wb.dictionary_freeze= True
X_name = wb.fit_transform(merge['name'])
del(wb)
X_name = X_name[:, np.array(np.clip(X_name.getnnz(axis=0) - 1, 0, 1), dtype=bool)]
print('[{}] Vectorize `name` completed.'.format(time.time() - start_time))

wb = CountVectorizer()
X_category1 = wb.fit_transform(merge['general_cat'])
X_category2 = wb.fit_transform(merge['subcat_1'])
X_category3 = wb.fit_transform(merge['subcat_2'])
print('[{}] Count vectorize `categories` completed.'.format(time.time() - start_time))

# wb= wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 3, "hash_ngrams_weights": [1.0, 1.0, 0.5],
wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.0, 1.0],
                                                              "hash_size": 2 ** 28, "norm": "l2", "tf": 1.0,
                                                              "idf": None})
                         , procs=8)
wb.dictionary_freeze= True
X_description = wb.fit_transform(merge['item_description'])
del(wb)
X_description = X_description[:, np.array(np.clip(X_description.getnnz(axis=0) - 1, 0, 1), dtype=bool)]
print('[{}] Vectorize `item_description` completed.'.format(time.time() - start_time))

lb = LabelBinarizer(sparse_output=True)
X_brand = lb.fit_transform(merge['brand_name'])
print('[{}] Label binarize `brand_name` completed.'.format(time.time() - start_time))

X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']],
                                      sparse=True).values)
print('[{}] Get dummies on `item_condition_id` and `shipping` completed.'.format(time.time() - start_time))
print(X_dummies.shape, X_description.shape, X_brand.shape, X_category1.shape, X_category2.shape, X_category3.shape,
      X_name.shape)
sparse_merge = hstack((X_dummies, X_description, X_brand, X_category1, X_category2, X_category3, X_name)).tocsr()

print('[{}] Create sparse merge completed'.format(time.time() - start_time))

# Remove features with document frequency <=1
print(sparse_merge.shape)
mask = np.array(np.clip(sparse_merge.getnnz(axis=0) - 1, 0, 1), dtype=bool)
sparse_merge = sparse_merge[:, mask]
X = sparse_merge[:nrow_train]
X_test = sparse_merge[nrow_test:]
print(sparse_merge.shape)

gc.collect()
train_X, train_y = X, y
if develop:
    train_X, valid_X, train_y, valid_y = train_test_split(X, y, train_size=0.90, random_state=233)

model = FM_FTRL(alpha=0.01, beta=0.01, L1=0.00001, L2=0.1, D=sparse_merge.shape[1], alpha_fm=0.01, L2_fm=0.0, init_fm=0.01,
                D_fm=200, e_noise=0.0001, iters=15, inv_link="identity", threads=4)

model.fit(train_X, train_y)
print('[{}] Train ridge v2 completed'.format(time.time() - start_time))
if develop:
    predsfm = model.predict(X=valid_X)
    print("FM_FTRL dev RMSLE:", rmsle(np.expm1(valid_y), np.expm1(predsfm))) # 0.44532 # Full data 0.424681

predsFM = model.predict(X_test)
print('[{}] Predict FM_FTRL completed'.format(time.time() - start_time))

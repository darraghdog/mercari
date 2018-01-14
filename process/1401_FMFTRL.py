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
from nltk.stem import PorterStemmer
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import lightgbm as lgb

#Add https://www.kaggle.com/anttip/wordbatch to your kernel Data Sources, 
#until Kaggle admins fix the wordbatch pip package installation
sys.path.insert(0, '../input/wordbatch/wordbatch/')
import wordbatch

from wordbatch.extractors import WordBag, WordHash
from wordbatch.models import FTRL, FM_FTRL

from nltk.corpus import stopwords
start_time = time.time()
from time import gmtime, strftime
import psutil


os.chdir('/home/darragh/mercari/data')

def cpuStats():
        print(sys.version)
        print(psutil.cpu_percent())
        print(psutil.virtual_memory())  # physical memory usage
        pid = os.getpid()
        py = psutil.Process(pid)
        memoryUse = py.memory_info()[0] / 2. ** 30
        print('memory GB:', memoryUse)
        
cpuStats()

def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y0), 2)))


def split_cat(text):
    try:
        return text.split("/")
    except:
        return ("No Label", "No Label", "No Label")


def handle_missing_inplace(dataset):
    dataset['category_name'].fillna(value='missing', inplace=True)
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
        
def normalize_text(text):
    return u" ".join(
        [x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] \
         if len(x) > 1 and x not in stopwords])

NUM_BRANDS = 4500
NUM_CATEGORIES = 1200

def getMerge():
    from nltk.corpus import stopwords
    train = pd.read_csv('../data/train.tsv', sep='\t', encoding='utf-8')
    test = pd.read_csv('../data/test.tsv', sep='\t', encoding='utf-8')
    glove_file = '../feat/glove.6B.50d.txt'
    threads = 8
    save_dir = '../feat'
    

    
    develop = False
    develop= True
        
    # Define helpers for text normalization
    stopwords = {x: 1 for x in stopwords.words('english')}
    non_alphanums = re.compile(u'[^A-Za-z0-9]+')
    

    
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
    merge['target'] = np.log1p(merge["price"])
    submission = test[['test_id']]
    
    #EXTRACT DEVELOPTMENT TEST
    trnidx, validx = train_test_split(range(train.shape[0]), random_state=233, train_size=0.90)
    
    #del train
    #del test
    gc.collect()
    return merge, trnidx, validx

merge, trnidx, validx = getMerge()
cpuStats()
merge.info(memory_usage='deep')


def processMerge(merge):
    merge['general_cat'], merge['subcat_1'], merge['subcat_2'] = \
        zip(*merge['category_name'].apply(lambda x: split_cat(x)))
    #merge.drop('category_name', axis=1, inplace=True)
    print('[{}] Split categories completed.'.format(time.time() - start_time))
    
    handle_missing_inplace(merge)
    print('[{}] Handle missing completed.'.format(time.time() - start_time))
    
    cutting(merge)
    print('[{}] Cut completed.'.format(time.time() - start_time))
    
    to_categorical(merge)
    print('[{}] Convert categorical completed'.format(time.time() - start_time))
    
    '''
    Crossed columns
    '''
    # my understanding on how to replicate what layers.crossed_column does. One
    # can read here: https://www.tensorflow.org/tutorials/linear.
    def cross_columns(x_cols):
        """simple helper to build the crossed columns in a pandas dataframe
        """
        crossed_columns = dict()
        colnames = ['_'.join(x_c) for x_c in x_cols]
        for cname, x_c in zip(colnames, x_cols):
            crossed_columns[cname] = x_c
        return crossed_columns
    
    merge['item_condition_id_str'] = merge['item_condition_id'].astype(str)
    merge['shipping_str'] = merge['shipping'].astype(str)
    x_cols = (
              ['brand_name',  'item_condition_id_str'],
              ['brand_name',  'subcat_1'],
              ['brand_name',  'subcat_2'],
              ['brand_name',  'general_cat'],
              #['brand_name',  'subcat_1',  'item_condition_id_str'],
              #['brand_name',  'subcat_2',  'item_condition_id_str'],
              #['brand_name',  'general_cat',  'item_condition_id_str'],
              ['brand_name',  'shipping_str'],
              ['shipping_str',  'item_condition_id_str'],
              ['shipping_str',  'subcat_2'],
              ['item_condition_id_str',  'subcat_2']          
              )
    crossed_columns_d = cross_columns(x_cols)
    categorical_columns = list(
        merge.select_dtypes(include=['object']).columns)
    
    D = 2**30
    for k, v in crossed_columns_d.items():
        print ('Crossed column ', k)
        outls_ = []
        indicator = 0 
        for col in v:
            outls_.append((np.array(merge[col].apply(hash)))%D + indicator)
            indicator += 10**6
        merge[k] = sum(outls_).tolist()
        
    return merge, crossed_columns_d
    
merge, crossed_columns_d = processMerge(merge)
cpuStats()
merge.info(memory_usage='deep')

'''
Count crossed cols
'''
cross_nm = [k for k in crossed_columns_d.keys()]
lb = LabelBinarizer(sparse_output=True)
x_col = lb.fit_transform(merge[cross_nm[0]])
for i in range(1, len(cross_nm)):
    x_col = hstack((x_col, lb.fit_transform(merge[cross_nm[i]])))
del(lb)
cpuStats()

    
'''
Hash name
'''
wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.5, 1.0],
                                                              "hash_size": 2 ** 29, "norm": None, "tf": 'binary',
                                                              "idf": None,
                                                              }), procs=8)
wb.dictionary_freeze= True
X_name = wb.fit_transform(merge['name'])
del(wb)
X_name = X_name[:, np.array(np.clip(X_name.getnnz(axis=0) - 1, 0, 1), dtype=bool)]
print('[{}] Vectorize `name` completed.'.format(time.time() - start_time))

cpuStats()
    

'''
Hash category
'''

wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.0, 1.0],
                                                              "hash_size": 2 ** 20, "norm": None, "tf": 'binary',
                                                              "idf": None,
                                                              }), procs=8)
wb.dictionary_freeze= True
cat = merge["category_name"].str.replace('/', ' ')
X_cat = wb.fit_transform(cat)
del(wb)
X_cat = X_cat[:, np.array(np.clip(X_cat.getnnz(axis=0) - 1, 0, 1), dtype=bool)]
print('[{}] Vectorize `category` completed.'.format(time.time() - start_time))
cpuStats()

'''
Count category
'''

wb = CountVectorizer()
X_category1 = wb.fit_transform(merge['general_cat'])
X_category2 = wb.fit_transform(merge['subcat_1'])
X_category3 = wb.fit_transform(merge['subcat_2'])
print('[{}] Count vectorize `categories` completed.'.format(time.time() - start_time))
cpuStats()

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
cpuStats()

lb = LabelBinarizer(sparse_output=True)
X_brand = lb.fit_transform(merge['brand_name'])
print('[{}] Label binarize `brand_name` completed.'.format(time.time() - start_time))
cpuStats()


crossed_columns_d.keys()
X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']],
                                      sparse=True).values)


merge[crossed_columns_d.keys()[1]].astype('category').head()
merge['subcat_1'].head()

print('[{}] Get dummies on `item_condition_id` and `shipping` completed.'.format(time.time() - start_time))
print(X_dummies.shape, X_description.shape, X_brand.shape, X_category1.shape, X_category2.shape, X_category3.shape,
      X_name.shape, X_cat.shape, x_col.shape)
sparse_merge = hstack((X_dummies, X_description, X_brand, X_category1, X_category2, X_category3, X_name, X_cat,
                       x_col)).tocsr()

print('[{}] Create sparse merge completed'.format(time.time() - start_time))

print(sparse_merge.shape)
mask = np.array(np.clip(sparse_merge.getnnz(axis=0) - 1, 0, 1), dtype=bool)
sparse_merge = sparse_merge[:, mask]
X = sparse_merge[:nrow_train]
X_test = sparse_merge[nrow_test:]
print(sparse_merge.shape)

gc.collect()
#train_X, train_y = X, y
if develop:
    #train_X1, valid_X1, train_y1, valid_y1 = train_test_split(X, y, train_size=0.90, random_state=233)
    train_X, valid_X, train_y, valid_y = X[trnidx], X[validx], y.values[trnidx], y.values[validx]

cpuStats()


model = FM_FTRL(alpha=0.01, beta=0.01, L1=0.00001, L2=0.1, D=sparse_merge.shape[1], alpha_fm=0.01, L2_fm=0.0, init_fm=0.01,
                D_fm=200, e_noise=0.0001, iters=1, inv_link="identity", threads=threads) #iters=15

baseline = 1.
for i in range(15):
    model.fit(train_X , train_y , verbose=1)
    predsfm = model.predict(X=valid_X)
    score_ = rmsle(np.expm1(valid_y), np.expm1(predsfm))
    print("FM_FTRL dev RMSLE:", score_)
    if score_ < baseline:
        baseline = score_
    else:
        break
    

print('[{}] Train ridge v2 completed'.format(time.time() - start_time))
if develop:
    predsfm = model.predict(X=valid_X)
    print("FM_FTRL dev RMSLE:", rmsle(np.expm1(valid_y), np.expm1(predsfm)))


predsFM = model.predict(X_test)
print('[{}] Predict FM_FTRL completed'.format(time.time() - start_time))
merge.info(memory_usage='deep')

#del X, y, X_test, train_X, valid_X, train_y, valid_y
gc.collect()


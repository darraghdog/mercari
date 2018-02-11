# encoding=utf8  
import sys  
#reload(sys)  
#sys.setdefaultencoding('utf8')
import os, math, gc, time, random, csv
start_time = time.time()
import numpy as np
from numba import jit
from collections import Counter
from scipy.sparse import csr_matrix, hstack, vstack
import nltk, re
from nltk.tokenize import ToktokTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
import tensorflow as tf
import multiprocessing as mp
def _get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)
_get_session()
from keras.layers import Input, Dropout, Dense, BatchNormalization, \
    Activation, concatenate, GRU, Embedding, Flatten, Bidirectional, \
    MaxPooling1D, Conv1D, Add, Reshape, Lambda
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping#, TensorBoard
from keras import backend as K
from keras import optimizers
from keras import initializers
from keras.preprocessing.sequence import pad_sequences
import warnings
warnings.simplefilter(action='ignore')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import psutil

#Add https://www.kaggle.com/anttip/wordbatch to your kernel Data Sources, 
#until Kaggle admins fix the wordbatch pip package installation
sys.path.insert(0, '../input/wordbatch/wordbatch/')
import wordbatch
from wordbatch.extractors import WordBag, WordHash
from wordbatch.models import FTRL, FM_FTRL
from time import gmtime, strftime


def cpuStats():
        print(sys.version)
        print(psutil.cpu_percent())
        print(psutil.virtual_memory())  # physical memory usage
        pid = os.getpid()
        py = psutil.Process(pid)
        memoryUse = py.memory_info()[0] / 2. ** 30
        print('memory GB:', memoryUse)

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

def prep_brand_name(df):
    # Add brand name to name
    ix = (df['brand_name']==df['brand_name']) & (~df['brand_name'].str.lower().fillna('ZZZZZZ').isin(df['name'].str.lower()))
    df['name'][ix] = df['brand_name'][ix] + ' ' + df['name'][ix]
    return df['name']

def cross_columns(x_cols):
    """simple helper to build the crossed columns in a pandas dataframe
    """
    crossed_columns = dict()
    colnames = ['_'.join(x_c) for x_c in x_cols]
    for cname, x_c in zip(colnames, x_cols):
        crossed_columns[cname] = x_c
    return crossed_columns

def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in props.columns:
        if props[col].dtype.name not in ['object', 'category']:  # Exclude strings
            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",props[col].dtype)
            
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all(): 
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)
            
            # Print new column type
            print("dtype after: ",props[col].dtype)
            print("******************************")
    
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props

x_cols = (
          ['brand_name',  'item_condition_id_str'],
          ['brand_name',  'subcat_1'],
          ['brand_name',  'subcat_2'],
          ['brand_name',  'general_cat'],
          ['brand_name',  'shipping_str'],
          ['shipping_str',  'item_condition_id_str'],
          ['shipping_str',  'subcat_2'],
          ['item_condition_id_str',  'subcat_2']          
          )

def normalize_text(text):
    return u" ".join(
        [x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] \
         if len(x) > 1 and x not in stopwords])


def prepSparseTrain(moddict, filename, rowidx = None):
    
    df = pd.read_csv(filename, sep='\t', encoding='utf-8')
    
    if rowidx!=None:
        df =  df.loc[rowidx]
    
    print('[{}] Finished to load data'.format(time.time() - start_time))
    print('Data shape: ', df.shape)    
    dftt = df[(df.price < 1.0)]
    df = df.drop(df[(df.price < 1.0)].index)    
    del dftt['price']
    nrow_df = df.shape[0]    
    y = np.log1p(df["price"])    
    df['target'] = np.log1p(df["price"])
    
    # Start prepping df
    df['name'] = prep_brand_name(df)
    df['general_cat'], df['subcat_1'], df['subcat_2'] = \
        zip(*df['category_name'].apply(lambda x: split_cat(x)))
    print('[{}] df split categories completed.'.format(time.time() - start_time))    
    handle_missing_inplace(df)
    print('[{}] Handle missing completed.'.format(time.time() - start_time))
    cutting(df)
    print('[{}] Cut completed.'.format(time.time() - start_time))
    to_categorical(df)
    print('[{}] Convert categorical completed'.format(time.time() - start_time))
    df['item_condition_id_str'] = df['item_condition_id'].astype(str)
    df['shipping_str'] = df['shipping'].astype(str)
    crossed_columns_d = cross_columns(x_cols)
    categorical_columns = list(
        df.select_dtypes(include=['object']).columns)
    
    D = 2**30
    for k, v in crossed_columns_d.items():
        print ('Crossed column ', k)
        outls_ = []
        indicator = 0 
        for col in v:
            outls_.append((np.array(df[col].apply(hash)))%D + indicator)
            indicator += 10**6
        df[k] = sum(outls_).tolist()
    cross_nm = [k for k in crossed_columns_d.keys()]
    moddict['cross_binarizer'] = {}
    moddict['cross_binarizer'][cross_nm[0]] = LabelBinarizer(sparse_output=True)
    moddict['cross_binarizer'][cross_nm[0]].fit(df[cross_nm[0]])
    X_col = moddict['cross_binarizer'][cross_nm[0]].transform(df[cross_nm[0]])
    for i in range(1, len(cross_nm)):
        moddict['cross_binarizer'][cross_nm[i]] = LabelBinarizer(sparse_output=True)
        moddict['cross_binarizer'][cross_nm[i]].fit(df[cross_nm[i]])
        X_col = hstack((X_col, moddict['cross_binarizer'][cross_nm[i]].transform(df[cross_nm[i]])))
        
    print('[{}] Finished cross column binarizer'.format(time.time() - start_time))

    moddict['wordbatch'] = {}
    moddict['wordbatch']['name'] = wordbatch.WordBatch(normalize_text, 
                                           extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.5, 1.0],
                                                                "hash_size": 2 ** 29, "norm": None, "tf": 'binary',
                                                                "idf": None, }), procs=4)
    moddict['wordbatch']['name'].dictionary_freeze= True
    moddict['wordbatch']['name'].fit(df['name'])
    X_name = moddict['wordbatch']['name'].transform(df['name'])
    moddict['wordbatch']['name_mask'] = np.array(np.clip(X_name.getnnz(axis=0) - 1, 0, 1), dtype=bool)
    X_name = X_name[:, moddict['wordbatch']['name_mask']]
    
    print('[{}] Vectorize `name` completed.'.format(time.time() - start_time))
    
    
    '''
    start_time = time.time()    
    moddict['wordbatch']['category'] = wordbatch.WordBatch(normalize_text, 
                                            extractor=(WordBag, {"hash_ngrams": 4, "hash_ngrams_weights": [1.0, 1.0, 1.0, 1.0],
                                                                  "hash_size": 2 ** 20, "norm": None, "tf": 'binary',
                                                                  "idf": None,}), procs=8)
    moddict['wordbatch']['category'].dictionary_freeze = True
    moddict['wordbatch']['category'].fit(df["category_name"].str.replace('/', ' '))
    X_cat = moddict['wordbatch']['category'].transform(df["category_name"].str.replace('/', ' '))
    moddict['wordbatch']['category_mask'] = np.array(np.clip(X_cat.getnnz(axis=0) - 1, 0, 1), dtype=bool)
    X_cat = X_cat[:, moddict['wordbatch']['category_mask']]
    print('[{}] Vectorize `category` completed.'.format(time.time() - start_time))
    '''
    
    moddict['wordbatch']['category'] = wordbatch.WordBatch(normalize_text, 
                                            extractor=(WordBag, {"hash_ngrams": 4, "hash_ngrams_weights": [1.0, 1.0, 1.0, 1.0],
                                                                  "hash_size": 2 ** 20, "norm": None, "tf": 'binary',
                                                                  "idf": None,}), procs=4)
    moddict['wordbatch']['category'].dictionary_freeze = True
    cats = pd.Series(categories).str.replace('/', ' ')
    moddict['wordbatch']['category'].fit(cats)
    X_cat_tmp = moddict['wordbatch']['category'].transform(cats)
    moddict['wordbatch']['categorydict'] = dict([(c, X_cat_tmp.getrow(row)) for (c, row) in zip(cats.tolist(), range(len(cats)))])
    X_cat = vstack(([moddict['wordbatch']['categorydict'][c] for c in df["category_name"].str.replace('/', ' ')]))
    moddict['wordbatch']['category_mask'] = np.array(np.clip(X_cat.getnnz(axis=0) - 1, 0, 1), dtype=bool)
    X_cat = X_cat[:, moddict['wordbatch']['category_mask']]
    print('[{}] Vectorize `category` completed.'.format(time.time() - start_time))
    
    moddict['count'] = {}
    for col in ['general_cat', 'subcat_1', 'subcat_2']:
        moddict['count'][col] = CountVectorizer()
        moddict['count'][col].fit(df[col])
    X_category1 = moddict['count']['general_cat'].transform(df['general_cat'])
    X_category2 = moddict['count']['subcat_1'].transform(df['subcat_1'])
    X_category3 = moddict['count']['subcat_2'].transform(df['subcat_2'])
    print('[{}] Count vectorize `categories` completed.'.format(time.time() - start_time))
    
    moddict['wordbatch']['item_description']  = wordbatch.WordBatch(
                                        normalize_text, extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.0, 1.0],
                                       "hash_size": 2 ** 28, "norm": "l2", "tf": 1.0,
                                       "idf": None}), procs=8)
    moddict['wordbatch']['item_description'].dictionary_freeze= True
    moddict['wordbatch']['item_description'].fit(df['item_description'])
    X_description = moddict['wordbatch']['item_description'].transform(df['item_description'])
    moddict['wordbatch']['item_description_mask'] = np.array(np.clip(X_description.getnnz(axis=0) - 1, 0, 1), dtype=bool)
    X_description = X_description[:, moddict['wordbatch']['item_description_mask'] ]
    print('[{}] Vectorize `item_description` completed.'.format(time.time() - start_time))
    
    
    moddict['binarizer'] = {}
    for col in ['brand_name', 'item_condition_id', 'shipping']:
        moddict['binarizer'][col] = LabelBinarizer(sparse_output=True)
        moddict['binarizer'][col].fit(df[col])
    X_brand = moddict['binarizer']['brand_name'].transform(df['brand_name'])
    X_cond  = moddict['binarizer']['item_condition_id'].transform(df['item_condition_id'])
    X_ship  = moddict['binarizer']['shipping'].transform(df['shipping'])
    print('[{}] Label binarize `brand_name` and `dummies` completed.'.format(time.time() - start_time))

    X_all = hstack((X_cond, X_ship, X_description, X_brand, X_category1, X_category2, X_category3, 
                    X_name, X_cat, X_col)).tocsr()
    
    return df, X_all, y, moddict

def prepSparseTest(moddict, filename, rowidx = None):
    
    df = pd.read_csv(filename, sep='\t', encoding='utf-8')
    
    if rowidx!=None:
        df =  df.loc[rowidx]
        if 'price' in df.columns:
            y = np.log1p(df["price"])    
            df['target'] = np.log1p(df["price"])
    
    # Start prepping df
    df['name'] = prep_brand_name(df)
    df['general_cat'], df['subcat_1'], df['subcat_2'] = \
        zip(*df['category_name'].apply(lambda x: split_cat(x)))
    print('[{}] df split categories completed.'.format(time.time() - start_time))    
    handle_missing_inplace(df)
    print('[{}] Handle missing completed.'.format(time.time() - start_time))
    cutting(df)
    print('[{}] Cut completed.'.format(time.time() - start_time))
    to_categorical(df)
    print('[{}] Convert categorical completed'.format(time.time() - start_time))
    df['item_condition_id_str'] = df['item_condition_id'].astype(str)
    df['shipping_str'] = df['shipping'].astype(str)
    crossed_columns_d = cross_columns(x_cols)
    categorical_columns = list(
        df.select_dtypes(include=['object']).columns)
    
    D = 2**30
    for k, v in crossed_columns_d.items():
        print ('Crossed column ', k)
        outls_ = []
        indicator = 0 
        for col in v:
            outls_.append((np.array(df[col].apply(hash)))%D + indicator)
            indicator += 10**6
        df[k] = sum(outls_).tolist()
    cross_nm = [k for k in crossed_columns_d.keys()]
    X_col = moddict['cross_binarizer'][cross_nm[0]].transform(df[cross_nm[0]])
    for i in range(1, len(cross_nm)):
        X_col = hstack((X_col, moddict['cross_binarizer'][cross_nm[i]].transform(df[cross_nm[i]])))
        
    print('[{}] Finished cross column binarizer'.format(time.time() - start_time))
    
    X_name = moddict['wordbatch']['name'].transform(df['name'])
    X_name = X_name[:, moddict['wordbatch']['name_mask']]
    print('[{}] Vectorize `name` completed.'.format(time.time() - start_time))
    
    '''
    X_cat = moddict['wordbatch']['category'].transform(df["category_name"].str.replace('/', ' '))
    X_cat = X_cat[:, moddict['wordbatch']['category_mask']]
    '''
    X_cat = vstack(([moddict['wordbatch']['categorydict'][c] for c in df["category_name"].str.replace('/', ' ')]))
    X_cat = X_cat[:, moddict['wordbatch']['category_mask']]
    print('[{}] Vectorize `category` completed.'.format(time.time() - start_time))
    
    X_category1 = moddict['count']['general_cat'].transform(df['general_cat'])
    X_category2 = moddict['count']['subcat_1'].transform(df['subcat_1'])
    X_category3 = moddict['count']['subcat_2'].transform(df['subcat_2'])
    print('[{}] Count vectorize `categories` completed.'.format(time.time() - start_time))
    
    X_description = moddict['wordbatch']['item_description'].transform(df['item_description'])
    X_description = X_description[:, moddict['wordbatch']['item_description_mask'] ]
    print('[{}] Vectorize `item_description` completed.'.format(time.time() - start_time))
    
    
    X_brand = moddict['binarizer']['brand_name'].transform(df['brand_name'])
    X_cond  = moddict['binarizer']['item_condition_id'].transform(df['item_condition_id'])
    X_ship  = moddict['binarizer']['shipping'].transform(df['shipping'])
    print('[{}] Label binarize `brand_name` and `dummies` completed.'.format(time.time() - start_time))
    
    print(X_cond.shape, X_ship.shape, X_description.shape, X_brand.shape, X_category1.shape, 
                   X_category2.shape, X_category3.shape, X_name.shape, X_cat.shape, X_col.shape)
    #((148254, 5), (148254, 1), (148254, 1443619), (148254, 4501), (148254, 14), 
    # (148254, 143), (148254, 961), (148254, 365715), (148254, 5305), (148254, 82586))
    
    #    print(X_dummies.shape, X_description.shape, X_brand.shape, X_category1.shape, X_category2.shape, X_category3.shape,
    #      X_name.shape, X_cat.shape, x_col.shape)
    #(2175894, 6) (2175894, 2040339) (2175894, 4501) (2175894, 14) (2175894, 143)
    #(2175894, 977) (2175894, 529016) (2175894, 2662) (2175894, 97140)

    X_all = hstack((X_cond, X_ship, X_description, X_brand, X_category1, X_category2, X_category3, 
                    X_name, X_cat, X_col)).tocsr()
    
    return df, X_all

def trainFMFTRL():
    # Load Data
    dftrain, X_train, y_train, moddict = prepSparseTrain({}, trn_file, trnidx)
    dfvalid, X_valid                   = prepSparseTest(moddict, trn_file, validx)
    # Train the model
    modelfm = FM_FTRL(alpha=0.01, beta=0.01, L1=0.00001, L2=0.1, D=X_train.shape[1], alpha_fm=0.01, L2_fm=0.0, init_fm=0.01,
                D_fm=200, e_noise=0.0001, iters=1, inv_link="identity", threads=threads) #iters=15
    baseline = 1.
    threshold = .0002
    for i in range(15):
        modelfm.fit(X_train , dftrain.target.values, verbose=1)
        predsfm = modelfm.predict(X=X_valid)
        score_ = rmsle(np.expm1(dfvalid.target.values), np.expm1(predsfm))
        print("FM_FTRL dev RMSLE:", score_)
        if score_ + threshold < baseline:
            baseline = score_
        else:
            break    
        # 0.42919 with zeros in val
        # 0.42160 removing zeros in val
        # X_train.shape = (1333501, 1902850)
        # ('FM_FTRL dev RMSLE:', 0.42850571762280409)
    
    # Reduce the number of columns
    keep_cols = ['train_id', 'name', 'item_condition_id', 'category_name', 'brand_name', 'price', \
                'shipping', 'item_description', 'target', 'general_cat', 'subcat_1', 'subcat_2']
    dftrain, dfvalid = dftrain[keep_cols], dfvalid[keep_cols]
    
    return dftrain, dfvalid, y_train, moddict, modelfm, predsfm

def predictFMFTRL():

    dftest , X_test = prepSparseTest(moddict, tst_file,  tstidx,)
    predsFM = modelfm.predict(X_test)
    print('[{}] Predict FM_FTRL completed'.format(time.time() - start_time))
        # Reduce the number of columns
    keep_cols = ['test_id', 'name', 'item_condition_id', 'category_name', 'brand_name', \
                'shipping', 'item_description', 'general_cat', 'subcat_1', 'subcat_2']
    dftest =  dftest[keep_cols]
    
    return dftest, predsFM
        
    

'''
To do 
Make the submission index for test
'''

'''
Set params
'''
print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
cpuStats()

develop = False
develop= True
glove_file = '../feat/glove.6B.50d.txt'
threads = 4
save_dir = '../feat'
os.chdir('/home/darragh/mercari/data')
trn_file = '../data/train.tsv'
tst_file = '../data/test.tsv'

'''
Load all categories in train and test
'''
# Define helpers for text normalization
stopwords = {x: 1 for x in stopwords.words('english')}
non_alphanums = re.compile(u'[^A-Za-z0-9]+')
NUM_BRANDS = 4500
NUM_CATEGORIES = 1200
categories = pd.concat([pd.read_table(trn_file, sep='\t', encoding='utf-8', usecols = [3]), \
                        pd.read_table(tst_file, sep='\t', encoding='utf-8', usecols = [3])]) \
                            ['category_name'].fillna(value='missing').unique()
cpuStats()
'''
Train and test split
'''
trnidx, validx = train_test_split(range(sum(1 for line in open(trn_file))-1), random_state=233, train_size=0.90)
tstidx = range(sum(1 for line in open(tst_file))-1)

'''
Train and predict FTRL
'''
dftrain, dfvalid, y_train, moddict, modelfm, predsfm = trainFMFTRL()
gc.collect()
cpuStats()
dftest, predsFM = predictFMFTRL()
gc.collect()
cpuStats()

'''
Join train and test from GRU ***To be changed
'''
train = pd.read_csv('../data/train.tsv', sep='\t', encoding='utf-8')
test = pd.read_csv('../data/test.tsv', sep='\t', encoding='utf-8')
y = np.log1p(train["price"])    
nrow_train = len(y)
train['target'] = np.log1p(train["price"])
merge = pd.concat([train, test])
del train, test
merge['name'] = prep_brand_name(merge)
merge['general_cat'], merge['subcat_1'], merge['subcat_2'] = \
        zip(*merge['category_name'].apply(lambda x: split_cat(x)))
print('[{}] df split categories completed.'.format(time.time() - start_time))    
handle_missing_inplace(merge)
print('[{}] Handle missing completed.'.format(time.time() - start_time))
cutting(merge)
print('[{}] Cut completed.'.format(time.time() - start_time))
to_categorical(merge)
print('[{}] Convert categorical completed'.format(time.time() - start_time))


'''
nrow_train = dftrain.shape[0]
nrow_valid = dfvalid.shape[0]
nrow_test  = dftest.shape[0]
merge = pd.concat([dftrain, dfvalid, dftest])
submission = dftest[['test_id']]
del dftrain
del dfvalid
del dftest
gc.collect()
cpuStats()
'''


'''
GRU
'''

tech_mapper = {
               'unblocked' : 'unlocked',
               'ipad3' : ' ipad3 ',
               'ipad2' : ' ipad2 ',
               'ipad1' : ' ipad1 ',
               'ipad 3' : ' ipad3 ',
               'ipad 2' : ' ipad2 ',
               'ipad 1' : ' ipad1 ',
               '8 gb'     : ' 8gb ',
               '16 gb'    : ' 16gb ',
               '64 gb'    : ' 64gb ',
               '256 gb'   : ' 256gb ',
               '500 gb'   : ' 500gb ',
               '32 gb'    : ' 32gb ',
               'bnwt'     : 'brand new with tags',
               'nwt'      : 'new with tags',
               'bnwot'    : 'brand new without tags',
               'bnwob'    : 'brand new without box',
               'nwot'     : 'new without tags',
               'bnip'     : 'brand new in packet',
               'nip'      : 'new in packet',
               'bnib'     : 'brand new in box',
               'nib'      : 'new in box',
               'mib'      : 'mint in box',
               'mwob:'    : 'mint without box',
               'mip'      : 'mint in packet',
               'mwop'     : 'mint without packet',
               } 

def replace_maps(sent):
    if sent!=sent:
        return sent
    for k, v in tech_mapper.items():
        sent = sent.replace(k, v)
    return sent

import multiprocessing as mp
pool = mp.Pool(processes=4)
for col in ['name', 'item_description']:
    merge[col] = merge[col].str.lower()
    merge[col] = pool.map(replace_maps, merge[col].values)
pool.close
print('[{}] Finished replacing text...'.format(time.time() - start_time))


print("Handling categorical variables...")
le = LabelEncoder()
merge['category'] = le.fit_transform(merge.category_name)

hi_brand_cts = merge['brand_name'][:nrow_train].value_counts()
hi_brand_cts = hi_brand_cts[hi_brand_cts>2].index.values
merge.brand_name[~merge.brand_name.isin(hi_brand_cts)] = '_lo_count_'
merge['brand'] = le.fit_transform(merge.brand_name)
del le



special_pattern = re.compile( 
    u"([\u0101-\ufffd])|"  
    u"([\x96-\xfc])" 
    "+", flags=re.UNICODE)

def remove_special(sent):
    if sent != sent:
        return sent
    return (special_pattern.sub(r' ', sent)) 

import multiprocessing as mp
pool = mp.Pool(processes=4)
for col in ["item_description", "name", "brand_name"]:
    print("Clean special characters from " + col)
    merge[col] = pool.map(remove_special, merge[col].values)
pool.close
print('[{}] Finished remove bogus characters...'.format(time.time() - start_time))

# Replace the category slash
merge["category_name_split"] = merge["category_name"].str.replace(' ', '_')
merge["category_name_split"] = merge["category_name_split"].str.replace('/', ' ')
print('[{}] Finished PROCESSING CATEGORICAL DATA...'.format(time.time() - start_time))


toktok = ToktokTokenizer()
tokSentMap = {}
def tokSent(sent):
    sent = sent.replace('/', ' ')
    return " ".join(toktok.tokenize(rgx.sub('', sent)))

rgx = re.compile('[%s]' % '!"#%&()*,-./:;<=>?@[\\]^_`{|}~\t\n')   

pool = mp.Pool(processes=4)
merge['description_token']      = pool.map(tokSent, merge['item_description'].str.lower().tolist())
merge['name_token']             = pool.map(tokSent, merge['name'].str.lower().tolist())
merge['category_token']         = pool.map(tokSent, merge['category_name'].str.lower().tolist())
merge['brand_token']            = pool.map(tokSent, merge['brand_name'].str.lower().tolist())
print('[{}] Finished Tokenizing text...'.format(time.time() - start_time))
pool.close

@jit
def list_flatten(var):
    list_ = []
    for sent_ in var:
        list_ += sent_.split(' ')
    return Counter(list_)

wordlist = []
for col in ['name_token', 'category_token', 'brand_token']:
    flat_counter = list_flatten(merge[[col]].values[:,0])
    wordlist += [k for (k, v) in flat_counter.items() if v>3]
    wordlist = list(set(wordlist))
wordlist = set(wordlist)

embeddings_matrix = []
embedding_map = {}
#f = open('../feat/wiki.en.vec')
f = open(glove_file)
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

# Get the dot product
def posn_to_sparse(dt, embedding_map):
    sprow = []
    spcol = []
    spdata = []
    for c, (nm, ct, bt) in enumerate(zip(dt['name_token'].values,
                               dt['category_token'].values,
                               dt['brand_token'].values)):
        sent = " ".join([nm, ct, bt])
        ids = [embedding_map[s] for s in sent.split(' ') if s in embedding_map]
        n_ = len(ids)
        sprow += [c]*n_
        spcol += ids
        spdata += [1]*n_    
    shape_ = (dt.shape[0], len(embedding_map.keys())) 
    dt_ids = csr_matrix((spdata, (sprow, spcol)), shape=shape_)
    return dt_ids

                 
@jit
def myTokenizerFitJit(strls, max_words = 25000, filt = True):
    list_=[]
    for sent in strls:
        if filt:
            sent = rgx.sub(' ', sent)
        for s in sent.split(' '):
            if s!= '':
                list_.append(s)
    return Counter(list_).most_common(max_words)

def myTokenizerFit(strls, max_words = 25000):
    mc = myTokenizerFitJit(strls, max_words = 25000)
    return dict((i, c+1) for (c, (i, ii)) in enumerate(mc))  

@jit
def fit_sequence(str_, tkn_, filt = True):
    labels = []
    for sent in str_:
        if filt:
            sent = rgx.sub(' ', sent)
        tk = []
        for i in sent.split(' '):
            if i in tkn_:
                if i != '':
                    tk.append(tkn_[i])
        labels.append(tk)
    return labels

tok_raw_cat = myTokenizerFit(merge.category_name_split[:nrow_train].str.lower().unique(), max_words = 800)
gc.collect()
tok_raw_nam = myTokenizerFit(merge.name[:nrow_train].str.lower().unique(), max_words = 25000)
gc.collect()
tok_raw_dsc = myTokenizerFit(merge.description_token[:nrow_train].str.lower().unique(), max_words = 25000)
gc.collect()
tok_raw_ntk = myTokenizerFit(merge.name_token[:nrow_train].str.lower().unique(), max_words = 50000)
gc.collect()
print('[{}] Finished FITTING TEXT DATA...'.format(time.time() - start_time))    
print("   Transforming text to seq...")
merge["seq_category_name_split"] =     fit_sequence(merge.category_name_split.str.lower(), tok_raw_cat)
gc.collect()
merge["seq_item_description"] =        fit_sequence(merge.description_token.str.lower(), tok_raw_dsc)
merge['seq_item_description_rev']      = [list(reversed(l)) for l in merge.seq_item_description]
gc.collect()
merge["seq_name"] =                    fit_sequence(merge.name.str.lower(), tok_raw_nam)
gc.collect()
merge["seq_name_token"] =              fit_sequence(merge.name_token.str.lower(), tok_raw_ntk, filt = False)
gc.collect()
print('[{}] Finished PROCESSING TEXT DATA...'.format(time.time() - start_time))
merge.head()
#EXTRACT DEVELOPTMENT TEST

# Make a sparse matrix of the ids of words
merge.reset_index(drop=True, inplace=True)
merge_ids = posn_to_sparse(merge, embedding_map)
# Get the dense layer input of the text
densemrg = merge_ids.dot(embeddings_matrix)#.todense()

mean_, sd_ = densemrg.mean(), densemrg.std()
densemrg -= mean_
densemrg /= sd_

print(merge.shape)
print(densemrg.shape)

#PROCESS TEXT: RAW
print("Text to seq process...")
print("   Fitting tokenizer...")

#EMBEDDINGS MAX VALUE
MAX_CAT = max(tok_raw_cat.values())+1
MAX_NAM = max(tok_raw_nam.values())+1
MAX_NTK = max(tok_raw_ntk.values())+1
MAX_DSC = max(tok_raw_dsc.values())+1
MAX_CATEGORY = np.max(merge.category.max())+1
MAX_BRAND = np.max(merge.brand.max())+1
merge.item_condition_id = merge.item_condition_id.astype(int)
MAX_CONDITION = np.max(merge.item_condition_id.astype(int).max())+1
    
def get_keras_data(dataset):
    X = {
        'name': pad_sequences(dataset.seq_name, 
                              maxlen=max([len(l) for l in dataset.seq_name]))
        ,'ntk': pad_sequences(dataset.seq_name_token, 
                              maxlen=max([len(l) for l in dataset.seq_name_token]))
        ,'item_desc': pad_sequences(dataset.seq_item_description, 
                              maxlen=max([len(l) for l in dataset.seq_item_description]))
        ,'item_desc_rev': pad_sequences(dataset.seq_item_description_rev, 
                              maxlen=max([len(l) for l in dataset.seq_item_description_rev]))
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

def len_argsort(seq):
	return sorted(range(len(seq)), key=lambda x: len(seq[x]))

def map_sort(seq1, seq2):
	return sorted(range(len(seq1)), key=lambda x: len(seq1[x])*100+len(seq2[x]))
    
def reset_data(dt, bsize):
    max_step = dt.shape[0]
    n_batches = int(np.ceil(max_step*1. / float(bsize)))
    batch_steps = np.array(random.sample(range(n_batches), n_batches))
    #sorted_ix = np.array(len_argsort(dt["seq_item_description"].tolist()))
    sorted_ix = np.array(map_sort(dt["seq_item_description"].tolist(), dt["seq_name_token"].tolist()))
    dt.reset_index(drop=True, inplace = True)  
    return max_step, batch_steps, sorted_ix, dt

def trn_generator(dn, dt, y, bsize):
    while True:
        max_step, batch_steps, sorted_ix, dt = reset_data(dt, bsize)
        for batch in batch_steps:
            from_ = batch*bsize
            to_   = min((batch+1)*bsize, max_step)
            ix_   = sorted_ix[from_:to_]
            Xbatch = dt.iloc[ix_]
            Xbatch = get_keras_data(Xbatch)
            Xbatch['dense_name'] = dn[ix_]
            ybatch = dt.target.iloc[ix_]
            yield Xbatch, ybatch

def val_generator(dn, dt, y, bsize):
    while 1:
        max_step, batch_steps, sorted_ix, dt = reset_data(dt, bsize)
        for batch in batch_steps:
            from_ = batch*bsize
            to_   = min((batch+1)*bsize, max_step)
            ix_   = sorted_ix[from_:to_]
            Xbatch = dt.iloc[ix_]
            Xbatch = get_keras_data(Xbatch)
            Xbatch['dense_name'] = dn[ix_]
            ybatch = dt.target.iloc[ix_]
            yield Xbatch, ybatch
            
def tst_generator(dn, dt, bsize):
    while 1:
        for batch in range(int(np.ceil(dt.shape[0]*1./bsize))):
        #for batch in range(dt.shape[0]/bsize+1):
            from_ = batch*bsize
            to_   = min((batch+1)*bsize, dt.shape[0])
            Xbatch = dt.iloc[from_:to_]
            Xbatch = get_keras_data(Xbatch)
            Xbatch['dense_name'] = dn[from_:to_]
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
    ntk = Input(shape=[None], name="ntk")
    item_desc = Input(shape=[None], name="item_desc")
    item_desc_rev = Input(shape=[None], name="item_desc_rev")
    category_name_split = Input(shape=[None], name="category_name_split")
    brand = Input(shape=[1], name="brand")
    item_condition = Input(shape=[1], name="item_condition")
    num_vars = Input(shape=[1], name="num_vars")
    dense_name = Input(shape=[densetrn.shape[1]], name="dense_name")
    
    #Embeddings layers
    emb_size = 60
    emb_name                = Embedding(MAX_NAM, emb_size//2)(name) 
    emb_ntk                 = Embedding(MAX_NTK, emb_size//2)(ntk) 
    
    emb_item_desc_vals      =  Embedding(MAX_DSC, emb_size//2)
    emb_item_desc           = emb_item_desc_vals (item_desc) 
    emb_item_desc_rev       = emb_item_desc_vals (item_desc_rev) 
    
    emb_category_name_split = Embedding(MAX_CAT, emb_size//3)(category_name_split) 
    emb_brand               = Embedding(MAX_BRAND, 8)(brand)
    emb_item_condition      = Embedding(MAX_CONDITION, 5)(item_condition)
    
    
    rnn_layer1 = GRU(16, recurrent_dropout=0.0) (emb_item_desc)
    rnn_layer2 = GRU(8, recurrent_dropout=0.0) (emb_category_name_split)
    rnn_layer3 = GRU(8, recurrent_dropout=0.0) (emb_name)
    rnn_layer4 = GRU(8, recurrent_dropout=0.0) (emb_ntk)
    rnn_layer6 = GRU(16, recurrent_dropout=0.0) (emb_item_desc_rev)
    
    dense_l = Dropout(dr*3)(Dense(256,activation='relu') (dense_name))
    dense_l = Dropout(dr*3)(Dense(32,activation='relu') (dense_name))
    
    #main layer
    main_l = concatenate([
        Flatten() (emb_brand)
        , Flatten() (emb_item_condition)
        , rnn_layer1
        , rnn_layer2
        , rnn_layer3
        , rnn_layer4
        , rnn_layer6
        , dense_l
        , num_vars
    ])
    main_l = Dropout(dr)(Dense(128,activation='relu') (main_l))
    main_l = Dropout(dr)(Dense(64,activation='relu') (main_l))
    
    #output
    output = Dense(1,activation="linear") (main_l)
    
    #model
    model = Model([name, brand, ntk, item_desc, dense_name, item_desc_rev
                   , category_name_split #,category
                   , item_condition, num_vars], output)
    optimizer = optimizers.Adam()
    model.compile(loss='mse', 
                  optimizer=optimizer)
    return model
    
print('[{}] Finished DEFINING MODEL...'.format(time.time() - start_time))


merge.reset_index(drop=True, inplace=True)
dtrain, dvalid, test = merge[:nrow_train].iloc[trnidx], merge[:nrow_train].iloc[validx], merge[nrow_train:]
densetrn, denseval, densetst = densemrg[:nrow_train][trnidx], densemrg[:nrow_train][validx], densemrg[nrow_train:]
#dtrain, dvalid, test = merge[:nrow_train], merge[nrow_train:(nrow_train+nrow_valid)], merge[-nrow_test:]
#densetrn, denseval, densetst = densemrg[:nrow_train], densemrg[nrow_train:(nrow_train+nrow_valid)], densemrg[-nrow_test:]
#del merge, densemrg
gc.collect()
cpuStats()

epochs = 2
batchSize = 512 * 4
steps = (dtrain.shape[0]/batchSize+1)*epochs
lr_init, lr_fin = 0.015, 0.012
lr_decay  = (lr_init - lr_fin)/steps
model = get_model()
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


val_sorted_ix = np.array(map_sort(dvalid["seq_item_description"].tolist(), dvalid["seq_name_token"].tolist()))
tst_sorted_ix = np.array(map_sort(test  ["seq_item_description"].tolist(), test  ["seq_name_token"].tolist()))
y_pred_epochs = []
yspred_epochs = []
for c, lr in enumerate([0.010, 0.009, 0.008]): # , 0.006, 0.007,
    K.set_value(model.optimizer.lr, lr)
    model.fit_generator(
                        trn_generator(densetrn, dtrain, dtrain.target, batchSize)
                        , epochs=1#,epochs
                        , max_queue_size=1
                        , steps_per_epoch = int(np.ceil(dtrain.shape[0]*1./batchSize))
                        , validation_data = val_generator(denseval, dvalid, dvalid.target, batchSize)
                        , validation_steps = int(np.ceil(dvalid.shape[0]*1./batchSize))
                        , verbose=2
                        )
    y_pred_epochs.append(model.predict_generator(
                    tst_generator(denseval[val_sorted_ix], dvalid.iloc[val_sorted_ix], batchSize)
                    , steps = int(np.ceil(dvalid.shape[0]*1./batchSize))
                    , max_queue_size=1 
                    , verbose=2)[val_sorted_ix.argsort()])
    yspred_epochs.append(model.predict_generator(
                    tst_generator(densetst[tst_sorted_ix], test.iloc[tst_sorted_ix], batchSize)
                    , steps = int(np.ceil(test.shape[0]*1./batchSize))
                    , max_queue_size=1 
                    , verbose=2)[tst_sorted_ix.argsort()])
    print("Epoch %s rmsle %s"%(epochs+c+1, eval_model(dvalid.price.values, y_pred_epochs[-1])))
y_pred = sum(y_pred_epochs)/len(y_pred_epochs)
yspred = sum(yspred_epochs)/len(yspred_epochs)
print("Bagged Epoch %s rmsle %s"%(epochs+c+1, eval_model(dvalid.price.values, y_pred)))
# 0.43359

merge.head()
# Bagged Epoch 5 rmsle 0.429088545511
print("Bagged FM & Nnet", rmsle(dvalid.price.values, (np.expm1(predsfm)*0.5 + np.expm1(y_pred[:,0])*0.5)  ))


bag_preds = np.expm1(predsFM)*0.5 + np.expm1(yspred[:,0])*0.5  
print('[{}] Finished predicting test set...'.format(time.time() - start_time))
submission = test[["test_id"]].astype(int)
submission["price"] = bag_preds
submission.to_csv("./myBag_1201.csv", index=False)
#submission.to_csv("./myBag"+log_subdir+"_{:.6}.csv".format(v_rmsle), index=False)
print('[{}] Finished submission...'.format(time.time() - start_time))
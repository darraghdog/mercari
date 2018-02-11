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
from scipy.sparse import csr_matrix, hstack, vstack
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
from keras.layers import GlobalMaxPooling1D
from keras import initializers
from keras.utils import plot_model
import warnings
warnings.simplefilter(action='ignore')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import psutil

#Add https://www.kaggle.com/anttip/wordbatch to your kernel Data Sources, 
#until Kaggle admins fix the wordbatch pip package installation
sys.path.insert(0, '../input/wordbatch/wordbatch/')
import wordbatch

from wordbatch.extractors import WordBag, WordHash
from wordbatch.models import FTRL, FM_FTRL

from nltk.corpus import stopwords
start_time = time.time()
from time import gmtime, strftime

def cpuStats():
        print(sys.version)
        print(psutil.cpu_percent())
        print(psutil.virtual_memory())  # physical memory usage
        pid = os.getpid()
        py = psutil.Process(pid)
        memoryUse = py.memory_info()[0] / 2. ** 30
        print('memory GB:', memoryUse)


'''
Params
'''
#os.chdir('/Users/dhanley2/Documents/mercari/data')
'''
os.chdir('/home/darragh/mercari/data')
glove_file = '../feat/glove.6B.50d.txt'
threads = 8
save_dir = '../feat'
trn_file = '../data/train.tsv'
tst_file = '../data/test.tsv'
cpuStats()
moddict = {}
mdir    = '../feat'

'''
from subprocess import check_output, call
print(check_output(["ls", "../input"]).decode("utf8"))
#print(check_output(["ls", "../input/glove-vectors"]).decode("utf8"))
print(check_output(["ls", "../input/glove.6b.50d"]).decode("utf8"))
print(check_output(["ls", "../input/mercari-price-suggestion-challenge"]).decode("utf8"))
glove_file = '../input/glove.6b.50d/glove.6B.50d.txt'
threads = 4
save_dir = '../feat'
trn_file = '../input/mercari-price-suggestion-challenge/train.tsv'
tst_file = '../input/mercari-price-suggestion-challenge/test.tsv'
mdir    = '.'


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

def cross_columns(x_cols):
    """simple helper to build the crossed columns in a pandas dataframe
    """
    crossed_columns = dict()
    colnames = ['_'.join(x_c) for x_c in x_cols]
    for cname, x_c in zip(colnames, x_cols):
        crossed_columns[cname] = x_c
    return crossed_columns

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

def eval_model(y_true, val_preds):
    val_preds = np.expm1(val_preds)
    y_pred = val_preds[:, 0]
    v_rmsle = rmsle(y_true, y_pred)
    print("RMSLE error on dev test: "+str(v_rmsle))
    return v_rmsle

def replace_maps(sent):
    if sent!=sent:
        return sent
    for k, v in tech_mapper.items():
        sent = sent.replace(k, v)
    return sent

def remove_special(sent):
    if sent != sent:
        return sent
    return (special_pattern.sub(r' ', sent)) 

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

# Get the dot product
def posn_to_sparse_dscr(dt, embedding_map):
    sprow = []
    spcol = []
    spdata = []
    for c, (ds) in enumerate(dt['item_description'].values):
        sent = ds[:500]
        ids = [embedding_map[s] for s in sent.split(' ') if s in embedding_map]
        n_ = len(ids)
        sprow += [c]*n_
        spcol += ids
        spdata += [1]*n_    
    shape_ = (dt.shape[0], len(embedding_map.keys())) 
    dt_ids = csr_matrix((spdata, (sprow, spcol)), shape=shape_)
    return dt_ids

# Get the dot product
def posn_to_sparse_name(dt, embedding_map):
    sprow = []
    spcol = []
    spdata = []
    for c, (nm) in enumerate(dt['name_token'].values):
        sent = nm
        ids = [embedding_map[s] for s in sent.split(' ') if s in embedding_map]
        n_ = len(ids)
        sprow += [c]*n_
        spcol += ids
        spdata += [1]*n_    
    shape_ = (dt.shape[0], len(embedding_map.keys())) 
    dt_ids = csr_matrix((spdata, (sprow, spcol)), shape=shape_)
    return dt_ids

# Get the dot product
def posn_to_sparse_cat(dt, embedding_map):
    sprow = []
    spcol = []
    spdata = []
    for c, (ct) in enumerate(dt['category_token'].values):
        sent = ct
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

#KERAS MODEL DEFINITION
def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 \
              for i, pred in enumerate(y_pred)]
    return (sum(to_sum) * (1.0/len(y))) ** 0.5

def normalize_text(text):
    return u" ".join(
        [x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] \
         if len(x) > 1 and x not in stopwords])

@jit
def list_flatten(var):
    list_ = []
    for sent_ in var:
        list_ += sent_.split(' ')
    return Counter(list_)

def prepFMFeatures(df):
    ix = (df['brand_name']==df['brand_name']) & \
                    (~df['brand_name'].str.lower().fillna('ZZZZZZ').isin(df['name'].str.lower()))
    df['name'][ix] = df['brand_name'][ix] + ' ' + df['name'][ix]
    cpuStats()
    gc.collect()
    
    df['general_cat'], df['subcat_1'], df['subcat_2'] = \
        zip(*df['category_name'].apply(lambda x: split_cat(x)))
    print('[{}] Split categories completed.'.format(time.time() - start_time))
    
    handle_missing_inplace(df)
    print('[{}] Handle missing completed.'.format(time.time() - start_time))
    
    cutting(df)
    print('[{}] Cut completed.'.format(time.time() - start_time))
    
    to_categorical(df)
    print('[{}] Convert categorical completed'.format(time.time() - start_time))
    cpuStats()   
    gc.collect()
    '''
    Crossed columns
    '''

    
    df['item_condition_id_str'] = df['item_condition_id'].astype(str)
    df['shipping_str'] = df['shipping'].astype(str)
    
    categorical_columns = list(
        df.select_dtypes(include=['object']).columns)
    
    D = 2**22
    for k, v in crossed_columns_d.items():
        print ('Crossed column ', k)
        outls_ = []
        indicator = 0 
        for col in v:
            outls_.append((np.array(df[col].apply(hash)))%D + indicator)
            indicator += 10**6
        df[k] = np.array(sum(outls_).tolist()).astype(np.int32)
    
    return df


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

# Define helpers for text normalization
stopwords = {x: 1 for x in stopwords.words('english')}
non_alphanums = re.compile(u'[^A-Za-z0-9]+')

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

crossed_columns_d = cross_columns(x_cols)

print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))    

categories = pd.concat([pd.read_table(trn_file, sep='\t', encoding='utf-8', usecols = [3]), \
                        pd.read_table(tst_file, sep='\t', encoding='utf-8', usecols = [3])]) \
                            ['category_name'].fillna(value='missing').unique()
categories = pd.Series(categories ).str.replace('/', ' ').values
cpuStats()

def trainFMFTRL(moddict):

    merge = pd.read_csv(trn_file, sep='\t', encoding='utf-8')
    #test = pd.read_csv(tst_file, sep='\t', encoding='utf-8')
    
    print('[{}] Finished to load data'.format(time.time() - start_time))
    print('Train shape: ', merge.shape)
    
    dftt = merge[(merge.price < 1.0)]
    merge = merge.drop(merge[(merge.price < 1.0)].index)
    del dftt['price']
    nrow_train = merge.shape[0]
    # print(nrow_train, nrow_test)
    y = np.log1p(merge["price"])
    merge = pd.concat([merge, dftt])
    merge['target'] = np.log1p(merge["price"])
    #EXTRACT DEVELOPTMENT TEST
    trnidx, validx = train_test_split(range(merge[:nrow_train].shape[0]), random_state=233, train_size=0.90)
    gc.collect()
    cpuStats()
    
    merge = prepFMFeatures(merge)
    cpuStats()
    merge.head()
    
    '''
    Count crossed cols
    '''
    cross_nm = [k for k in crossed_columns_d.keys()]
    moddict['cross_cols'] = {}
    for i in range(0, len(cross_nm)):
        moddict['cross_cols'][cross_nm[i]] = LabelBinarizer(sparse_output=True)
        moddict['cross_cols'][cross_nm[i]].fit(merge[cross_nm[i]])
        if i == 0:
            x_col = moddict['cross_cols'][cross_nm[i]].transform(merge[cross_nm[i]])
        else:
            x_col = hstack((x_col, moddict['cross_cols'][cross_nm[i]].fit_transform(merge[cross_nm[i]])))
        del merge[cross_nm[i]]
    gc.collect()
    cpuStats()
    
    '''
    Hash name
    '''
    moddict['wb_name'] = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.5, 1.0],
                                                                  "hash_size": 2 ** 29, "norm": None, "tf": 'binary',
                                                                  "idf": None, 'verbose' : 1,
                                                                  }), procs=8)
    moddict['wb_name'].dictionary_freeze= True    
    X_name = moddict['wb_name'].fit_transform(merge['name'])
    moddict['wb_name_mask'] = np.where(X_name[:nrow_train].getnnz(axis=0) > 0)[0]
    X_name = X_name[:, moddict['wb_name_mask']]
    print('[{}] Vectorize `name` completed.'.format(time.time() - start_time))
    
    '''
    Hash category #2
    '''    
    moddict['wb_cat'] = wordbatch.WordBatch(normalize_text, 
                                            extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.0, 1.0],
                                                                  "hash_size": 2 ** 20, "norm": None, "tf": 'binary',
                                                                  "idf": None,}), procs=4)
    moddict['wb_cat'].dictionary_freeze = True
    ### This must be the full dataset
    #cats = merge["category_name"].str.replace('/', ' ').unique()
    moddict['wb_cat'].fit(categories)
    X_cat_tmp = moddict['wb_cat'].transform(categories)
    moddict['wb_cat_dict'] = dict([(c, X_cat_tmp.getrow(row)) for (c, row) in zip(categories.tolist(), range(len(categories)))])
    X_cat = vstack(([moddict['wb_cat_dict'][c] for c in merge["category_name"].str.replace('/', ' ')]))
    #moddict['wb_cat_mask'] = np.array(np.clip(X_cat[:nrow_train].getnnz(axis=0) - 1, 0, 1), dtype=bool)
    moddict['wb_cat_mask'] = np.where(X_cat[:nrow_train].getnnz(axis=0) > 0)[0]
    X_cat = X_cat[:, moddict['wb_cat_mask'] ]
    print('[{}] Vectorize `category` completed.'.format(time.time() - start_time))
    
    '''
    Count category
    '''
    
    moddict['wb_cat_ctgc'] = CountVectorizer()
    moddict['wb_cat_ctgc'].fit(merge['general_cat'])
    X_category1 = moddict['wb_cat_ctgc'].transform(merge['general_cat'])
    moddict['wb_cat_ctsc1'] = CountVectorizer()
    moddict['wb_cat_ctsc1'].fit(merge['subcat_1'])
    X_category2 = moddict['wb_cat_ctsc1'].transform(merge['subcat_1'])
    moddict['wb_cat_ctsc2'] = CountVectorizer()
    moddict['wb_cat_ctsc2'].fit(merge['subcat_2'])
    X_category3 = moddict['wb_cat_ctsc2'].transform(merge['subcat_2'])
    print('[{}] Count vectorize `categories` completed.'.format(time.time() - start_time))
    
    moddict['wb_dscr'] = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.0, 0.6],
                                                                  "hash_size": 2 ** 28, "norm": None, "tf": 'binary',
                                                                  "idf": None})
                                                                    , procs=8)
    moddict['wb_dscr'].dictionary_freeze= True
    X_description = moddict['wb_dscr'].fit_transform(merge['name'] + ' ' + merge['item_description'])
    #moddict['wb_dscr_mask'] = np.array(np.clip(X_description[:nrow_train].getnnz(axis=0) - 1, 0, 1), dtype=bool)
    moddict['wb_dscr_mask'] = np.where(X_description[:nrow_train].getnnz(axis=0) > 1)[0]
    X_description = X_description[:, moddict['wb_dscr_mask']]
    print('[{}] Vectorize `item_description` completed.'.format(time.time() - start_time))
    
    moddict['wb_brandname'] = LabelBinarizer(sparse_output=True)
    moddict['wb_brandname'].fit(merge['brand_name'][:nrow_train])
    X_brand = moddict['wb_brandname'].transform(merge['brand_name'])
    print('[{}] Label binarize `brand_name` completed.'.format(time.time() - start_time))
    
    moddict['wb_itemcond'] = LabelBinarizer(sparse_output=True)
    moddict['wb_itemcond'].fit(merge['item_condition_id'][:nrow_train])
    X_itemcond = moddict['wb_itemcond'].transform(merge['item_condition_id'])
    print('[{}] Label binarize `item_condition_id` completed.'.format(time.time() - start_time))    
    
    moddict['wb_shipping'] = LabelBinarizer(sparse_output=True)
    moddict['wb_shipping'].fit(merge['shipping'][:nrow_train])
    X_shipping = moddict['wb_shipping'].transform(merge['shipping'])
    print('[{}] Label binarize `shipping` completed.'.format(time.time() - start_time))    
    
    print(X_itemcond.shape, X_shipping.shape,  #X_dummies.shape, 
          X_description.shape, X_brand.shape, X_category1.shape, X_category2.shape, X_category3.shape,
          X_name.shape, X_cat.shape, x_col.shape)
    sparse_merge = hstack((X_itemcond, X_shipping, #X_dummies, 
                           X_description, X_brand, X_category1, X_category2, X_category3, X_name, X_cat,
                           x_col)).tocsr()
    
    print('[{}] Create sparse merge completed'.format(time.time() - start_time))
    
    print (50*'-')
    cpuStats()
    print (50*'-')
    # Remove features with document frequency <=1
    print(sparse_merge.shape)
    gc.collect()
    sparse_merge, y = sparse_merge[:nrow_train], y[:nrow_train]
    if develop:
        train_X, valid_X, train_y, valid_y = sparse_merge[trnidx], \
                                        sparse_merge[validx], \
                                        y.values[trnidx], y.values[validx]
        del sparse_merge
        gc.collect()
    print (50*'*')
    cpuStats()
    print (50*'*')
    print (train_X.shape[1])
    model = FM_FTRL(alpha=0.01, beta=0.01, L1=0.00001, L2=0.1, D=train_X.shape[1], alpha_fm=0.01, L2_fm=0.0, init_fm=0.01,
                    D_fm=200, e_noise=0.0001, iters=1, inv_link="identity", threads=4) #iters=15
    
    print (50*'|')
    cpuStats()
    print (50*'|')
    baseline = 1.
    for i in range(15):
        print (50*'-')
        cpuStats()
        print (50*'-')
        model.fit(train_X , train_y , verbose=1)
        predsfm = model.predict(X=valid_X)
        score_ = rmsle(np.expm1(valid_y), np.expm1(predsfm))
        print("FM_FTRL dev RMSLE:", score_)
        if score_ < baseline - 0.0004:
            baseline = score_
        else:
            break
    
    moddict['FMmodel'] = model
    
    print('[{}] Train ridge v2 completed'.format(time.time() - start_time))
    if develop:
        predsfm = moddict['FMmodel'].predict(X=valid_X)
        print("FM_FTRL dev RMSLE:", rmsle(np.expm1(valid_y), np.expm1(predsfm)))
    gc.collect()
    
    return merge, moddict, trnidx, validx, nrow_train, predsfm
    
def predictFMFTRL(mergetst):
    #mergetst = pd.read_csv(tst_file, sep='\t', encoding='utf-8')    
    
    print('[{}] Finished to load data'.format(time.time() - start_time))
    nrow_test = mergetst .shape[0]  # -dftt.shape[0]
    submission = mergetst[['test_id']]
    mergetst = prepFMFeatures(mergetst)
    cpuStats()
    mergetst.head()
    
    '''
    Count crossed cols
    '''
    cross_nm = [k for k in crossed_columns_d.keys()]
    for i in range(0, len(cross_nm)):
        if i == 0:
            x_col = moddict['cross_cols'][cross_nm[i]].transform(mergetst[cross_nm[i]])
        else:
            x_col = hstack((x_col, moddict['cross_cols'][cross_nm[i]].fit_transform(mergetst[cross_nm[i]])))
        del mergetst[cross_nm[i]]
    gc.collect()
    cpuStats()
    
    X_name = moddict['wb_name'].transform(mergetst['name'])
    X_name = X_name[:, moddict['wb_name_mask']]
    print('[{}] Vectorize `name` completed.'.format(time.time() - start_time))

    X_cat = vstack(([moddict['wb_cat_dict'][c] for c in mergetst["category_name"].str.replace('/', ' ')]))
    X_cat = X_cat[:, moddict['wb_cat_mask'] ]
    print('[{}] Vectorize `category` completed.'.format(time.time() - start_time))
    
    X_category1 = moddict['wb_cat_ctgc'].transform(mergetst['general_cat'])
    X_category2 = moddict['wb_cat_ctsc1'].transform(mergetst['subcat_1'])
    X_category3 = moddict['wb_cat_ctsc2'].transform(mergetst['subcat_2'])
    print('[{}] Count vectorize `categories` completed.'.format(time.time() - start_time))
    
    X_description = moddict['wb_dscr'].transform(mergetst['name'] + ' ' + mergetst['item_description'])
    X_description = X_description[:, moddict['wb_dscr_mask']]
    print('[{}] Vectorize `item_description` completed.'.format(time.time() - start_time))
    
    X_brand = moddict['wb_brandname'].transform(mergetst['brand_name'])
    print('[{}] Label binarize `brand_name` completed.'.format(time.time() - start_time))
    
    X_itemcond = moddict['wb_itemcond'].transform(mergetst['item_condition_id'])
    print('[{}] Label binarize `item_condition_id` completed.'.format(time.time() - start_time))    
    
    X_shipping = moddict['wb_shipping'].transform(mergetst['shipping'])
    print('[{}] Label binarize `shipping` completed.'.format(time.time() - start_time))
    
    print(50*'-')
    cpuStats()
    print(50*'-')
    print(X_itemcond.shape, X_shipping.shape,  #X_dummies.shape, 
          X_description.shape, X_brand.shape, X_category1.shape, X_category2.shape, X_category3.shape,
          X_name.shape, X_cat.shape, x_col.shape)
    X_test = hstack((X_itemcond, X_shipping, #X_dummies, 
                           X_description, X_brand, X_category1, X_category2, X_category3, X_name, X_cat,
                           x_col)).tocsr()
    print(50*'|')
    cpuStats()
    print(50*'|')
    
    print('[{}] Create sparse merge completed'.format(time.time() - start_time))
    
    predsFM = moddict['FMmodel'].predict(X_test)
    print('[{}] Predict FM_FTRL completed'.format(time.time() - start_time))
    print(50*'-')
    cpuStats()
    print(50*'-')
    return [mergetst, predsFM]


def prepGRUdf(df):                 
    pool = mp.Pool(processes=4)
    for col in ['name', 'item_description']:
        df[col] = df[col].str.lower()
        df[col] = pool.map(replace_maps, df[col].values)
    pool.close
    print('[{}] Finished replacing text...'.format(time.time() - start_time))
    cpuStats()
    
    df.brand_name[~df.brand_name.isin(hi_brand_cts)] = '_lo_count_'
    df['brand'] = le_brand.transform(df.brand_name)    
    
    pool = mp.Pool(processes=4)
    for col in ["item_description", "name", "brand_name"]:
        print("Clean special characters from " + col)
        df[col] = pool.map(remove_special, df[col].values)
    pool.close
    print('[{}] Finished remove bogus characters...'.format(time.time() - start_time))
    
    # Replace the category slash
    df["category_name_split"] = df["category_name"].str.replace(' ', '_')
    df["category_name_split"] = df["category_name_split"].str.replace('/', ' ')
    print('[{}] Finished PROCESSING CATEGORICAL DATA...'.format(time.time() - start_time))
    cpuStats()
    
    pool = mp.Pool(processes=4)
    df['description_token']      = pool.map(tokSent, df['item_description'].str.lower().tolist())
    df['name_token']             = pool.map(tokSent, df['name'].str.lower().tolist())
    df['category_token']         = pool.map(tokSent, df['category_name'].str.lower().tolist())
    df['brand_token']            = pool.map(tokSent, df['brand_name'].str.lower().tolist())
    print('[{}] Finished Tokenizing text...'.format(time.time() - start_time))
    pool.close
    cpuStats()
    
    return df

def seqTokenDf(df):
    
    print('[{}] Finished FITTING TEXT DATA...'.format(time.time() - start_time))    
    print("   Transforming text to seq...")
    df["seq_category_name_split"] =     fit_sequence(df.category_name_split.str.lower(), tok_raw_cat)
    gc.collect()
    df["seq_item_description"] =        fit_sequence(df.description_token.str.lower(), tok_raw_dsc)
    df['seq_item_description_rev']      = [list(reversed(l)) for l in df.seq_item_description]
    gc.collect()
    df["seq_name"] =                    fit_sequence(df.name.str.lower(), tok_raw_nam)
    gc.collect()
    df["seq_name_token"] =              fit_sequence(df.name_token.str.lower(), tok_raw_ntk, filt = False)
    gc.collect()
    print('[{}] Finished PROCESSING TEXT DATA...'.format(time.time() - start_time))
    cpuStats()
    return df

'''
Start the training 
'''
moddict = {}
mergetrn, moddict, trnidx, validx, nrow_train, predsfm = trainFMFTRL(moddict)
gc.collect()
cpuStats()

'''
Chunked inference
'''
with open(tst_file) as f:
   testlines = sum(1 for _ in f)
tstls = []
for c, chunk in enumerate(pd.read_csv(tst_file, chunksize=testlines//4, sep='\t', encoding='utf-8')):
    tstls.append(predictFMFTRL(chunk))

del moddict
gc.collect()
cpuStats()

'''
GRU
'''
toktok = ToktokTokenizer()
porter = PorterStemmer()
tokSentMap = {}
def tokSent(sent):
    sent = sent.replace('/', ' ')
    return " ".join(toktok.tokenize(rgx.sub('', sent)))

special_pattern = re.compile( 
    u"([\u0101-\ufffd])|"  
    u"([\x96-\xfc])" 
    "+", flags=re.UNICODE)
rgx = re.compile('[%s]' % '!"#%&()*,-./:;<=>?@[\\]^_`{|}~\t\n')  

hi_brand_cts = mergetrn['brand_name'].value_counts()[(mergetrn['brand_name'].value_counts()>2)].index.tolist()
le_brand = LabelEncoder()
le_brand.fit(hi_brand_cts+['_lo_count_'])

predsFM  =  np.concatenate([pred for (df, pred) in tstls])
tstls =  [prepGRUdf(df) for (df, pred) in tstls]
mergetst = pd.concat([df for df in tstls])


print('[{}] Finished predicting test set...'.format(time.time() - start_time))
submission = mergetst[["test_id"]].astype(int)
submission["price"] = np.expm1(predsFM)
submission.to_csv("./myFTRL_0502.csv", index=False)
print('[{}] Finished submission...'.format(time.time() - start_time))
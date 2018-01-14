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


os.chdir('/home/darragh/mercari/data')
#os.chdir('/Users/dhanley2/Documents/mercari/data')


train = pd.read_csv('../data/train.tsv', sep='\t', encoding='utf-8')
test = pd.read_csv('../data/test.tsv', sep='\t', encoding='utf-8')
glove_file = '../feat/glove.6B.50d.txt'
threads = 8
save_dir = '../feat'

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
merge['target'] = np.log1p(merge["price"])
submission = test[['test_id']]

#EXTRACT DEVELOPTMENT TEST
trnidx, validx = train_test_split(range(train.shape[0]), random_state=233, train_size=0.90)

#del train
#del test
gc.collect()

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

model = FM_FTRL(alpha=0.01, beta=0.01, L1=0.00001, L2=0.1, D=sparse_merge.shape[1], alpha_fm=0.01, L2_fm=0.0, init_fm=0.01,
                D_fm=200, e_noise=0.0001, iters=15, inv_link="identity", threads=threads) #iters=15
model.fit(train_X , train_y , verbose=1)

'''
modeltmp = FM_FTRL(alpha=0.01, beta=0.01, L1=0.00001, L2=0.1, D=sparse_merge.shape[1], alpha_fm=0.01, L2_fm=0.0, init_fm=0.01,
                D_fm=200, e_noise=0.0001, iters=1, inv_link="identity", threads=threads) #iters=15
for _ in  range(15):
    modeltmp.fit(train_X , train_y , verbose=1)
    predsfm = modeltmp.predict(X=valid_X)
    print("FM_FTRL dev RMSLE:", rmsle(np.expm1(valid_y), np.expm1(predsfm)))
    # Full data 0.424681
'''

print('[{}] Train ridge v2 completed'.format(time.time() - start_time))
if develop:
    predsfm = model.predict(X=valid_X)
    print("FM_FTRL dev RMSLE:", rmsle(np.expm1(valid_y), np.expm1(predsfm)))
    # Full data 0.424681

predsFM = model.predict(X_test)
print('[{}] Predict FM_FTRL completed'.format(time.time() - start_time))

'''
Save the objects to memory
'''

print('[{}] Start saving objects ... '.format(time.time() - start_time))

from six.moves import cPickle
f = open(os.path.join(save_dir, '_FMFTRL.save'), 'wb')
cPickle.dump({'tst': predsFM, 'val' : predsfm}, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

f = open(os.path.join(save_dir, '_Index.save'), 'wb')
cPickle.dump({'tst': trnidx, 'val' : validx}, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

f = open(os.path.join(save_dir, '_Wordbag.save'), 'wb')
cPickle.dump({'X_description': X_description, 'X_name' : X_name}, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

del X, y, X_test, train_X, valid_X, train_y, valid_y
del X_dummies, X_description, X_brand, X_category1, X_category2, X_category3, X_name
del mask, train, test, dftt
gc.collect()

print('[{}] Finished saving objects ... '.format(time.time() - start_time))

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

'''
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
'''

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
porter = PorterStemmer()
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
tok_raw_nam = myTokenizerFit(merge.name[:nrow_train].str.lower().unique(), max_words = 25000)
tok_raw_dsc = myTokenizerFit(merge.description_token[:nrow_train].str.lower().unique(), max_words = 25000)
tok_raw_ntk = myTokenizerFit(merge.name_token[:nrow_train].str.lower().unique(), max_words = 50000)
print('[{}] Finished FITTING TEXT DATA...'.format(time.time() - start_time))    
print("   Transforming text to seq...")
merge["seq_category_name_split"] =     fit_sequence(merge.category_name_split.str.lower(), tok_raw_cat)
merge["seq_item_description"] =        fit_sequence(merge.description_token.str.lower(), tok_raw_dsc)
merge["seq_name"] =                    fit_sequence(merge.name.str.lower(), tok_raw_nam)
merge["seq_name_token"] =              fit_sequence(merge.name_token.str.lower(), tok_raw_ntk, filt = False)
print('[{}] Finished PROCESSING TEXT DATA...'.format(time.time() - start_time))
merge.head()
#EXTRACT DEVELOPTMENT TEST

dtrain_ids = posn_to_sparse(merge[:nrow_train].iloc[trnidx], embedding_map)
dvalid_ids = posn_to_sparse(merge[:nrow_train].iloc[validx], embedding_map)
test_ids   = posn_to_sparse(merge[nrow_test:], embedding_map)

# Get the dense layer input of the text
densetrn = dtrain_ids.dot(embeddings_matrix)
denseval = dvalid_ids.dot(embeddings_matrix)
densetst = test_ids.dot(embeddings_matrix)
mean_, sd_ = densetrn.mean(), densetrn.std()
densetrn -= mean_
denseval -= mean_
densetst -= mean_
densetrn /= sd_
denseval /= sd_
densetst /= sd_

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
    category_name_split = Input(shape=[None], name="category_name_split")
    brand = Input(shape=[1], name="brand")
    item_condition = Input(shape=[1], name="item_condition")
    num_vars = Input(shape=[1], name="num_vars")
    dense_name = Input(shape=[densetrn.shape[1]], name="dense_name")
    
    #Embeddings layers
    emb_size = 60
    emb_name                = Embedding(MAX_NAM, emb_size//2)(name) 
    emb_ntk                 = Embedding(MAX_NTK, emb_size//2)(ntk) 
    emb_item_desc           = Embedding(MAX_DSC, emb_size)(item_desc) 
    emb_category_name_split = Embedding(MAX_CAT, emb_size//3)(category_name_split) 
    emb_brand               = Embedding(MAX_BRAND, 8)(brand)
    emb_item_condition      = Embedding(MAX_CONDITION, 5)(item_condition)
    
    rnn_layer1 = GRU(16, recurrent_dropout=0.0) (emb_item_desc)
    rnn_layer2 = GRU(8, recurrent_dropout=0.0) (emb_category_name_split)
    rnn_layer3 = GRU(8, recurrent_dropout=0.0) (emb_name)
    rnn_layer4 = GRU(8, recurrent_dropout=0.0) (emb_ntk)
    
    
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
        , dense_l
        , num_vars
    ])
    main_l = Dropout(dr)(Dense(128,activation='relu') (main_l))
    main_l = Dropout(dr)(Dense(64,activation='relu') (main_l))
    
    #output
    output = Dense(1,activation="linear") (main_l)
    
    #model
    model = Model([name, brand, ntk, item_desc, dense_name
                   , category_name_split #,category
                   , item_condition, num_vars], output)
    optimizer = optimizers.Adam()
    model.compile(loss='mse', 
                  optimizer=optimizer)
    return model

print('[{}] Finished DEFINING MODEL...'.format(time.time() - start_time))

merge.reset_index(drop=True, inplace=True)
dtrain = merge[:nrow_train].iloc[trnidx]
dvalid = merge[:nrow_train].iloc[validx]
test = merge[nrow_test:]

print('[{}] Merge done...'.format(time.time() - start_time))
del merge
gc.collect()

print('[{}] Delete done...'.format(time.time() - start_time))


epochs = 2
batchSize = 512 * 4
steps = (dtrain.shape[0]/batchSize+1)*epochs
lr_init, lr_fin = 0.015, 0.012
lr_decay  = (lr_init - lr_fin)/steps
model = get_model()
K.set_value(model.optimizer.lr, lr_init)
K.set_value(model.optimizer.decay, lr_decay)

print('[{}] Start modelling...'.format(time.time() - start_time))


model.fit_generator(
                    trn_generator(densetrn, dtrain, dtrain.target, batchSize)
                    , epochs=epochs
                    , max_queue_size=1
                    , steps_per_epoch = int(np.ceil(dtrain.shape[0]*1./batchSize))
                    , validation_data = val_generator(denseval, dvalid, dvalid.target, batchSize)
                    , validation_steps = int(np.ceil(dvalid.shape[0]*1./batchSize))
                    , verbose=1
                    )


val_sorted_ix = np.array(map_sort(dvalid["seq_item_description"].tolist(), dvalid["seq_name_token"].tolist()))
tst_sorted_ix = np.array(map_sort(test  ["seq_item_description"].tolist(), test  ["seq_name_token"].tolist()))
y_pred_epochs = []
yspred_epochs = []
for c, lr in enumerate([0.010, 0.009, 0.008]):#, 0.006]): # 0.007,
    K.set_value(model.optimizer.lr, lr)
    model.fit_generator(
                        trn_generator(densetrn, dtrain, dtrain.target, batchSize)
                        , epochs=1#,epochs
                        , max_queue_size=1
                        , steps_per_epoch = int(np.ceil(dtrain.shape[0]*1./batchSize))
                        , validation_data = val_generator(denseval, dvalid, dvalid.target, batchSize)
                        , validation_steps = int(np.ceil(dvalid.shape[0]*1./batchSize))
                        , verbose=1
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
# Bagged Epoch 5 rmsle 0.429088545511
print("Bagged FM & Nnet", rmsle(dvalid.price.values, np.expm1(predsfm)*0.5 + np.expm1(y_pred[:,0])*0.5  ))

f = open(os.path.join(save_dir, '_GRU.save'), 'wb')
cPickle.dump({'tst': yspred, 'val' : y_pred}, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

del densetrn, denseval, densetst, embeddings_matrix
gc.collect()

'''
Start the lightgbm
'''
def col2sparse(var, max_col):
    row = []
    col = []
    data = []
    for c, l_ in enumerate(var):
        n_ = len(l_)
        row += [c]*n_
        col += l_
        data += [1]*n_
    shape_ = (len(var), max_col+1)
    return csr_matrix((data, (row, col)), shape=shape_)

llcols = [("seq_category_name_split", MAX_CAT), ("seq_item_description", MAX_DSC), \
          ("seq_name_token", MAX_NTK)]
lcols = ["brand", "item_condition_id", "shipping", "category"]

spmatval = hstack([col2sparse(dvalid[c_].tolist(), max_col = max_val) for (c_, max_val) in llcols] + \
                  [col2sparse([[l] for l in dvalid[c_].tolist()], \
                               max_col = max(dtrain[c_].tolist())+1) for c_ in lcols]).tocsr().astype(np.float32)
spmattrn = hstack([col2sparse(dtrain[c_].tolist(), max_col = max_val) for (c_, max_val) in llcols] + \
                  [col2sparse([[l] for l in dtrain[c_].tolist()], \
                               max_col = max(dtrain[c_].tolist())+1) for c_ in lcols]).tocsr().astype(np.float32)
spmattst = hstack([col2sparse(test [c_].tolist(), max_col = max_val) for (c_, max_val) in llcols] + \
                  [col2sparse([[l] for l in test [c_].tolist()], \
                               max_col = max(dtrain[c_].tolist())+1) for c_ in lcols]).tocsr().astype(np.float32)
print(spmatval.shape)
print(spmattst.shape)
print(spmattrn.shape)

# Feature Engineering - Bayes Mean and count
n_folds = 2 
folds = np.array([random.randint(1,n_folds) for i in range(dtrain.shape[0])])

def bayesMean(dt_in, dt_out, t_col = "brand", y_col = "target"):
    mean_dict  = pd.groupby(dt_in[[t_col, y_col]], t_col).mean().to_dict()[y_col]
    ct_dict   = pd.groupby(dt_in[[t_col, y_col]], t_col).count().to_dict()[y_col]
    glbmean  = dt_in[y_col].values.mean()
    def bMeanSngl(vc, vm, glbmean = glbmean, prior = 5):
        return ((vc*vm)+(prior*glbmean))/(vc+prior)
    bmean_dict = dict((kc, bMeanSngl(vc, vm)) for ((kc, vc), (km, vm)) in \
                      zip(ct_dict.items(), mean_dict.items()))
    out = dt_out[t_col].apply(lambda x : bmean_dict.get(x, glbmean)).values
    return out

dtrain.head()
bcols = ["brand", "category"]
trn_bayes_mean = np.zeros((dtrain.shape[0], len(bcols)), dtype=float)
val_bayes_mean = np.zeros((dvalid.shape[0], len(bcols)), dtype=float)
tst_bayes_mean = np.zeros((test.shape[0], len(bcols)), dtype=float)

# Fill in out training set
for i in range(len(bcols)):
    for f in range(n_folds):
        idx_ = folds == 1+f
        trn_bayes_mean[~idx_, i] = bayesMean(dtrain[idx_], dtrain[~idx_], t_col = bcols[i])
# Now fill in valid and test set
for i in range(len(bcols)):
    val_bayes_mean[:, i] = bayesMean(dtrain, dvalid, t_col = bcols[i])
    tst_bayes_mean[:, i] = bayesMean(dtrain, test, t_col = bcols[i])
    

spmatval = hstack([spmatval, csr_matrix(spmatval.sum(axis=1)), csr_matrix(val_bayes_mean)])
spmattrn = hstack([spmattrn, csr_matrix(spmattrn.sum(axis=1)), csr_matrix(trn_bayes_mean)])
spmattst = hstack([spmattst, csr_matrix(spmattst.sum(axis=1)), csr_matrix(tst_bayes_mean)])

d_train = lgb.Dataset(spmattrn, label=dtrain.target)#, max_bin=1024)
d_valid = lgb.Dataset(spmatval, label=dvalid.target)#, max_bin=1024)
d_test  = lgb.Dataset(spmattst)#, max_bin=8192)
watchlist = [d_train, d_valid]

params = {
    'learning_rate': 0.75,
    'application': 'regression',
    'max_depth': 3,
    'num_leaves': 99,
    'verbosity': -1,
    'metric': 'RMSE',
    'nthread': threads,
    'max_bin' : 1024
}

modellgb = lgb.train(params, train_set=d_train, num_boost_round=5000, valid_sets=watchlist, \
early_stopping_rounds=500, verbose_eval=250) 

y_predlgb = modellgb.predict(spmatval)
y_predlgb = np.expand_dims(y_predlgb, 1)
yspredlgb = modellgb.predict(spmattst)
yspredlgb = np.expand_dims(yspredlgb, 1)

print("LGB trees rmsle %s"%( eval_model(dvalid.price.values, y_predlgb)))
# LGB trees 0 rmsle 0.446724284216

f = open(os.path.join(save_dir, '_LGB.save'), 'wb')
cPickle.dump({'tst': yspredlgb, 'val' : y_predlgb}, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()


'''
Bagging
'''
def cPload(fname):   
    FILE = open(fname, 'r')
    infile = cPickle.load(FILE)
    FILE.close()
    return infile    

y_predgru = cPload(os.path.join(save_dir, '_GRU.save'))
y_predfmf = cPload(os.path.join(save_dir, '_FMFTRL.save'))
y_predlgb = cPload(os.path.join(save_dir, '_LGB.save'))

y_predval = np.expm1(y_predfmf['val'])*0.4 + np.expm1(y_predgru['val'][:,0])*0.4  + np.expm1(y_predlgb['val'][:,0])*0.2 
print("Bagged rmsle %s"%( rmsle(dvalid.price.values, (y_predval))))
# Bagged rmsle 0.410522131994

bag_preds = np.expm1(y_predfmf['tst'])*0.4 + np.expm1(y_predgru['tst'][:,0])*0.4  + np.expm1(y_predlgb['tst'][:,0])*0.2 


'''
Stacking
'''

mat = np.transpose(np.vstack([y_predfmf['val'], y_predgru['val'][:,0], y_predlgb['val'][:,0]]))
mat = np.column_stack((mat, dvalid.shipping.values))
mat = np.column_stack((mat, mat[:,0]*mat[:,1], mat[:,0]*mat[:,2], mat[:,2]*mat[:,1]))
mat = np.column_stack((mat, mat[:,0]-mat[:,1], mat[:,0]-mat[:,2], mat[:,2]-mat[:,1]))


matval, yval = mat[75000:], dvalid.target.values[75000:]
mattrn, ytrn = mat[:75000], dvalid.target.values[:75000]

dstrain = lgb.Dataset(mattrn, label=ytrn)#, max_bin=1024)
dsvalid = lgb.Dataset(matval, label=yval)#, max_bin=1024)
dstest  = lgb.Dataset(matval)#, max_bin=8192)
watchlist = [dstrain, dsvalid]

params = {
    'learning_rate': .01,
    'application': 'regression',
    'max_depth': 3,
    'verbosity': -1,
    'metric': 'RMSE',
    'nthread': threads,
}

modelstk = lgb.train(params, train_set=dstrain, num_boost_round=1000, valid_sets=watchlist, \
early_stopping_rounds=50, verbose_eval=100) 
print("Bagged rmsle %s"%( rmsle(dvalid.price.values[75000:], (y_predval[75000:]))))




print('[{}] Finished predicting test set...'.format(time.time() - start_time))
submission = test[["test_id"]].astype(int)
submission["price"] = bag_preds
submission.to_csv("./myBag_1301.csv", index=False)
#submission.to_csv("./myBag"+log_subdir+"_{:.6}.csv".format(v_rmsle), index=False)
print('[{}] Finished submission...'.format(time.time() - start_time))


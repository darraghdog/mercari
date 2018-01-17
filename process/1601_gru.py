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
import psutil

#Add https://www.kaggle.com/anttip/wordbatch to your kernel Data Sources, 
#until Kaggle admins fix the wordbatch pip package installation
#sys.path.insert(0, '/Users/dhanley2/Documents/Wordbatch')
sys.path.insert(0, '../input/wordbatch/wordbatch/')
import wordbatch
from wordbatch.extractors import WordBag, WordHash
from wordbatch.models import FTRL, FM_FTRL

from nltk.corpus import stopwords
start_time = time.time()
from time import gmtime, strftime



NUM_BRANDS = 4500
NUM_CATEGORIES = 1200

develop = False
develop= True

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


# Define helpers for text normalization
stopwords = {x: 1 for x in stopwords.words('english')}
non_alphanums = re.compile(u'[^A-Za-z0-9]+')


def normalize_text(text):
    return u" ".join(
        [x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] \
         if len(x) > 1 and x not in stopwords])

print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))

cpuStats()

def getFMFTRL():
    #os.chdir('/Users/dhanley2/Documents/mercari/data')
    os.chdir('/home/darragh/mercari/data')
    train = pd.read_csv('../data/train.tsv', sep='\t', encoding='utf-8')
    test = pd.read_csv('../data/test.tsv', sep='\t', encoding='utf-8')
    glove_file = '../feat/glove.6B.50d.txt'
    threads = 4
    save_dir = '../feat'
       
    
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
    ix = (merge['brand_name']==merge['brand_name']) & (~merge['brand_name'].str.lower().fillna('ZZZZZZ').isin(merge['name'].str.lower()))
    merge['name'][ix] = merge['brand_name'][ix] + ' ' +merge['name'][ix]
    
    #EXTRACT DEVELOPTMENT TEST
    trnidx, validx = train_test_split(range(train.shape[0]), random_state=233, train_size=0.90)
    
    del train
    del test
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
    
    return merge, trnidx, validx, nrow_train, nrow_test, glove_file


merge, trnidx, validx, nrow_train, nrow_test, glove_file = getFMFTRL()


cpuStats()   
gc.collect()
cpuStats()

'''
fasttext mats
'''
fonm = open('ftext_name.txt','w')
#fods = open('ftext_dscr.txt','w')
for nm, ct, ds in zip(merge.name.tolist(), merge.category_name.str.replace('/', ' ').tolist(),
                      merge.item_description.tolist()):
    fonm.write('%s %s\n'%(ct.encode('ascii', 'ignore'),
                                 nm.encode('ascii', 'ignore')))
#    fods.write('%s %s %s\n'%(ct.encode('ascii', 'ignore'),
#                                 nm.encode('ascii', 'ignore'),
#                                 ds.encode('ascii', 'ignore')))
fonm.close()
#fods.close()

from pyfasttext import FastText
import fasttext
from tqdm import tqdm 


print('[{}] Start fasttext training'.format(time.time() - start_time))
model = fasttext.cbow('ftext_name.txt', 'model', dim=32, ws = 5, lr = .1, min_count  = 1, thread = 4, epoch = 10, silent=0)
modelcb = FastText('model.bin')
print('[{}] Start fasttext mat creation'.format(time.time() - start_time))

ftmat = np.zeros((merge.shape[0], 32))
for c, vals in tqdm(enumerate(merge[['category_name', 'name']].values)):
    ftmat[c] = modelcb.get_numpy_sentence_vector('%s %s'%(vals[0].replace('/', ' '), vals[1]))
ftmat = pd.DataFrame(ftmat)
print('[{}] Finished fasttext mat creation'.format(time.time() - start_time))
ftmat.head()

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
porter = PorterStemmer()
tokSentMap = {}
def tokSent(sent):
    sent = sent.replace('/', ' ')
    return " ".join(toktok.tokenize(rgx.sub('', sent)))

rgx = re.compile('[%s]' % '!"#%&()*,-./:;<=>?@[\\]^_`{|}~\t\n')   
                 
cpuStats()   

pool = mp.Pool(processes=4)
merge['description_token']      = pool.map(tokSent, merge['item_description'].str.lower().tolist())
merge['name_token']             = pool.map(tokSent, merge['name'].str.lower().tolist())
merge['category_token']         = pool.map(tokSent, merge['category_name'].str.lower().tolist())
merge['brand_token']            = pool.map(tokSent, merge['brand_name'].str.lower().tolist())
print('[{}] Finished Tokenizing text...'.format(time.time() - start_time))
pool.close

cpuStats()   

@jit
def list_flatten(var):
    list_ = []
    for sent_ in var:
        list_ += sent_.split(' ')
    return Counter(list_)

'''
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
'''

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

cpuStats()

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
gc.collect()
merge["seq_name"] =                    fit_sequence(merge.name.str.lower(), tok_raw_nam)
gc.collect()
merge["seq_name_token"] =              fit_sequence(merge.name_token.str.lower(), tok_raw_ntk, filt = False)
gc.collect()
print('[{}] Finished PROCESSING TEXT DATA...'.format(time.time() - start_time))
merge.head()
#EXTRACT DEVELOPTMENT TEST
cpuStats() 

'''
# Make a sparse matrix of the ids of words
merge_ids = posn_to_sparse(merge, embedding_map)
# Get the dense layer input of the text
densemrg = merge_ids.dot(embeddings_matrix)#.todense()

mean_, sd_ = densemrg.mean(), densemrg.std()
densemrg -= mean_
densemrg /= sd_

print(merge.shape)
print(densemrg.shape)
cpuStats() 
gc.collect()
'''
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
    
    
    #dense_l = Dropout(dr*3)(Dense(256,activation='relu') (dense_name))
    #dense_l = Dropout(dr)(Dense(32,activation='relu') (dense_name))
    
    #main layer
    main_l = concatenate([
        Flatten() (emb_brand)
        , Flatten() (emb_item_condition)
        , rnn_layer1
        , rnn_layer2
        , rnn_layer3
        , rnn_layer4
        , dense_name
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

cpuStats()
merge.reset_index(drop=True, inplace=True)
dtrain, dvalid, test = merge[:nrow_train].iloc[trnidx], merge[:nrow_train].iloc[validx], merge[nrow_test:]
# densetrn, denseval, densetst = densemrg[:nrow_train][trnidx], densemrg[:nrow_train][validx], densemrg[nrow_test:]
densetrn, denseval, densetst = ftmat.values[:nrow_train][trnidx], ftmat.values[:nrow_train][validx], ftmat.values[nrow_test:]
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
    cpuStats()
y_pred = sum(y_pred_epochs)/len(y_pred_epochs)
yspred = sum(yspred_epochs)/len(yspred_epochs)
print("Bagged Epoch %s rmsle %s"%(epochs+c+1, eval_model(dvalid.price.values, y_pred)))


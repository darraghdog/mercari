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

os.chdir('/home/darragh/mercari/data')
glove_file = '../feat/glove.6B.50d.txt'
threads = 8
save_dir = '../feat'
trn_file = '../data/train.tsv'
tst_file = '../data/test.tsv'
cpuStats()
moddict = {}
mdir    = '../feat'
train_size = 0.90
test_chunk_split = 5
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
train_size = 0.98
test_chunk_split = 2
'''


NUM_BRANDS = 4500
NUM_CATEGORIES = 1200

develop = False
develop= True

def printSection(str_):
    print('*'*70)
    print('%s %s %s'%('*'*(34-int(0.5*len(str_))), str_, '*'*(34-int(0.5*len(str_)))))
    print('*'*70)

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
    trnidx, validx = train_test_split(range(merge[:nrow_train].shape[0]), random_state=233, train_size=train_size)
    gc.collect()
    cpuStats()
    
    merge = prepFMFeatures(merge)
    cpuStats()
    merge.head()
        
    return merge, trnidx, validx, nrow_train
    
def predictFMFTRL(mergetst):
    #mergetst = pd.read_csv(tst_file, sep='\t', encoding='utf-8')    
    
    print('[{}] Finished to load data'.format(time.time() - start_time))
    nrow_test = mergetst .shape[0]  # -dftt.shape[0]
    submission = mergetst[['test_id']]
    mergetst = prepFMFeatures(mergetst)
    cpuStats()
    mergetst.head()
    
    return mergetst


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


def seqTokenStopWDrop(df):
    
    df['seq_category_name_split']  = [[item for item in l if item not in tok_raw_cat_drop] for l in df.seq_category_name_split]
    df['seq_item_description']     = [[item for item in l if item not in tok_raw_dsc_drop] for l in df.seq_item_description]
    df['seq_item_description_rev'] = [[item for item in l if item not in tok_raw_dsc_drop] for l in df.seq_item_description_rev]
    df['seq_name']                 = [[item for item in l if item not in tok_raw_nam_drop] for l in df.seq_name ]
    df['seq_name_token']           = [[item for item in l if item not in tok_raw_ntk_drop] for l in df.seq_name_token]
    cpuStats()
    
    return df


'''
Start the training 
'''
printSection('Start the FTRL training')
moddict = {}
mergetrn, trnidx, validx, nrow_train = trainFMFTRL(moddict)
gc.collect()
cpuStats()

'''
Chunked inference
'''
printSection('Predict the FTRL training')

with open(tst_file) as f:
   testlines = sum(1 for _ in f)
tstls = []
for c, chunk in enumerate(pd.read_csv(tst_file, chunksize=testlines//test_chunk_split, sep='\t', encoding='utf-8')):
    tstls.append(predictFMFTRL(chunk))

del moddict
gc.collect()
cpuStats()

'''
GRU
'''
printSection('Prep RNN features')

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


mergetrn = prepGRUdf(mergetrn)
#predsFM  =  np.concatenate([pred for (df, pred) in tstls])
tstls =  [prepGRUdf(df) for (df) in tstls]

# Get the tokens per word
tok_raw_cat = myTokenizerFit(mergetrn.category_name_split[:nrow_train].str.lower().unique(), max_words = 800); gc.collect()
tok_raw_nam = myTokenizerFit(mergetrn.name[:nrow_train].str.lower().unique(), max_words = 25000); gc.collect()
tok_raw_dsc = myTokenizerFit(mergetrn.description_token[:nrow_train].str.lower().unique(), max_words = 25000); gc.collect()
tok_raw_ntk = myTokenizerFit(mergetrn.name_token[:nrow_train].str.lower().unique(), max_words = 50000); gc.collect()


stoppers = set(stopwords.keys())
tok_raw_cat_drop = set([v for (k, v) in tok_raw_cat.items() if len(k)==1 or k in stoppers])
tok_raw_nam_drop = set([v for (k, v) in tok_raw_nam.items() if len(k)==1 or k in stoppers])
tok_raw_dsc_drop = set([v for (k, v) in tok_raw_dsc.items() if len(k)==1 or k in stoppers])
tok_raw_ntk_drop = set([v for (k, v) in tok_raw_ntk.items() if len(k)==1 or k in stoppers])

mergetrn = seqTokenDf(mergetrn)
tstls =  [seqTokenDf(df) for (df) in tstls]
nrow_test = sum([df.shape[0] for (df) in tstls])
gc.collect()

'''
Pretrained embeddings
'''
printSection('Load the Pretrained embeddings')


wordlist = []
for col in ['name_token', 'category_token']:
    flat_counter = list_flatten(mergetrn[[col]].values[:,0])
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
embeddings_matrix = np.array(embeddings_matrix, dtype='float16')
del wordlist


'''
Embeddings to dense
'''
# # Make a sparse matrix of the ids of words
# merge_ids = posn_to_sparse(mergetrn[:500000], embedding_map)
# # Get the dense layer input of the text
# densemrg = merge_ids.dot(embeddings_matrix)* 1./(merge_ids.sum(axis=1)+.5).astype('float16')
emb_mean_, emb_sd_  = np.float16(0.0107), np.float16(0.4188) # densemrg.mean(), densemrg.std()
# del merge_ids, densemrg
gc.collect()

def mergeEmbeddings(mergedf):
    merge_ids = posn_to_sparse(mergedf, embedding_map)
    densedf = merge_ids.dot(embeddings_matrix)* 1./(merge_ids.sum(axis=1)+.5).astype('float16')
    densedf -= emb_mean_
    densedf /= emb_sd_
    return densedf.astype('float16')


#PROCESS TEXT: RAW
print("Text to seq process...")
print("   Fitting tokenizer...")

#EMBEDDINGS MAX VALUE
MAX_NAM = max(tok_raw_nam.values())+1
MAX_NTK = max(tok_raw_ntk.values())+1
MAX_DSC = max(tok_raw_dsc.values())+1
MAX_CAT = max(tok_raw_cat.values())+1
MAX_BRAND = np.max(mergetrn.brand.max())+1
mergetrn.item_condition_id = mergetrn.item_condition_id.astype(int)
MAX_CONDITION = np.max(mergetrn.item_condition_id.astype(int).max())+1
    
def get_keras_data(dataset):
    X = {
        'name': pad_sequences(dataset.seq_name, 
                              maxlen=max(1, max([len(l) for l in dataset.seq_name])))
        ,'ntk': pad_sequences(dataset.seq_name_token, 
                              maxlen=max(1, max([len(l) for l in dataset.seq_name_token])))
        ,'item_desc': pad_sequences(dataset.seq_item_description, 
                              maxlen=max(1, max([len(l) for l in dataset.seq_item_description])))
        ,'item_desc_rev': pad_sequences(dataset.seq_item_description_rev, 
                              maxlen=max(1, max([len(l) for l in dataset.seq_item_description_rev])))
        ,'brand': np.array(dataset.brand)
        ,'category_name_split': pad_sequences(dataset.seq_category_name_split, 
                              maxlen=max(1, max([len(l) for l in dataset.seq_category_name_split])))
        ,'item_condition': np.array(dataset.item_condition_id)
        ,'num_vars': np.array(dataset[["shipping"]])
    }
    return X   

##############################################################################################################
##############################################################################################################


dr = 0.1

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
    
    dense_l = Dropout(dr*1)(Dense(256,activation='relu') (dense_name))
    
    dense_l = Dropout(dr*1)(Dense(32,activation='relu') (dense_name))
    dense_l = BatchNormalization()(dense_l)
    
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
    main_l = BatchNormalization()(main_l)
    main_l = Dropout(dr)(Dense(64,activation='relu') (main_l))
    main_l = BatchNormalization()(main_l)
    
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

printSection('Start the RNN training #1A')


dtrain   = mergetrn[:nrow_train].iloc[trnidx]
dvalid   = mergetrn[:nrow_train].iloc[validx]
del mergetrn
gc.collect()

def make_denstrn():
    # Break it up to save memory
    densetrn_part1 = mergeEmbeddings(dtrain[:70000]) # densetrain[:nrow_train][trnidx] 
    densetrn_part2 = mergeEmbeddings(dtrain[70000:]) # densetrain[:nrow_train][trnidx] 
    densetrn = np.concatenate([densetrn_part1, densetrn_part2])
    return densetrn

densetrn = make_denstrn()
denseval = mergeEmbeddings(dvalid) # densetrain[:nrow_train][validx]
gc.collect()


bags = 2
y_pred_epochs = []
session_conf = tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=1, device_count = {'CPU': 4})
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


for b in range(bags):
    #if b>0:
    #    dtrain = seqTokenStopWDrop(dtrain)
    #    dvalid = seqTokenStopWDrop(dvalid)
    #    gc.collect()
    val_sorted_ix = np.array(map_sort(dvalid["seq_item_description"].tolist(), dvalid["seq_name_token"].tolist()))
    printSection('Start the RNN training part1 bag%s'%(b))
    epochs = 2
    batchSize = 512 * 4
    steps = (dtrain.shape[0]/batchSize+1)*epochs
    lr_init, lr_fin = 0.015, 0.012
    lr_decay  = (lr_init - lr_fin)/steps
    model = get_model()
    K.set_value(model.optimizer.lr, lr_init)
    K.set_value(model.optimizer.decay, lr_decay)
    for i in range(epochs):
        model.fit_generator(
                            trn_generator(densetrn, dtrain, dtrain.target, batchSize)
                            , epochs=1
                            , max_queue_size=4
                            #, workers=4#, use_multiprocessing=True
                            , steps_per_epoch = int(np.ceil(dtrain.shape[0]*1.*0.9/batchSize))
                            , validation_data = val_generator(denseval, dvalid, dvalid.target, batchSize)
                            , validation_steps = int(np.ceil(dvalid.shape[0]*1.*0.9/batchSize))
                            , verbose=1
                            )
    
    from keras.models import load_model
    cpuStats()
    
    '''
    Training Keras model
    '''
    printSection('Start the RNN training part2 bag%s'%(b))
    lr_ls = [0.010, 0.009, 0.008]
    for c, lr in enumerate(lr_ls): # , 0.006, 0.007,
        K.set_value(model.optimizer.lr, lr)
        model.fit_generator(
                            trn_generator(densetrn, dtrain, dtrain.target, batchSize)
                            , epochs=1#,epochs
                            , max_queue_size=4
                            , steps_per_epoch = int(np.ceil(dtrain.shape[0]*1.*0.4/batchSize))
                            , validation_data = val_generator(denseval, dvalid, dvalid.target, batchSize)
                            , validation_steps = int(np.ceil(dvalid.shape[0]*1./batchSize))
                            , verbose=2
                            )
        y_pred_epochs.append(model.predict_generator(
                        tst_generator(denseval[val_sorted_ix], dvalid.iloc[val_sorted_ix], batchSize)
                        , steps = int(np.ceil(dvalid.shape[0]*1./batchSize))
                        , max_queue_size=1 
                        , verbose=2)[val_sorted_ix.argsort()])
        model.save_weights('%s/rnn_model%s.h5'%(mdir, c+(b*len(lr_ls))))
        print("Epoch %s rmsle %s"%(epochs+c+1, eval_model(dvalid.price.values, y_pred_epochs[-1])))
        cpuStats()
        
y_pred = sum(y_pred_epochs)/len(y_pred_epochs)
print("Bagged Epoch %s rmsle %s"%(epochs+c+1, eval_model(dvalid.price.values, y_pred)))
print("Bagged FM & Nnet", rmsle(dvalid.price.values, np.expm1(predsfm)*0.5 + np.expm1(y_pred[:,0])*0.5  ))

#del densetrn, denseval, dtrain, dvalid
gc.collect()
cpuStats()




##############################################################################################################
##############################################################################################################

dr = 0.1

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
    emb_name                = Embedding(MAX_NAM, 20)(name) 
    emb_ntk                 = Embedding(MAX_NTK, 20)(ntk) 
    emb_item_desc           = Embedding(MAX_DSC, 20) (item_desc) 
    emb_category_name_split = Embedding(MAX_CAT, 20)(category_name_split) 
    emb_brand               = Embedding(MAX_BRAND, 8)(brand)
    emb_item_condition      = Embedding(MAX_CONDITION, 5)(item_condition)
    
    
    #rnn_layer1 = GRU(16, recurrent_dropout=0.0) (emb_item_desc)
    rnn_layer1 = GlobalMaxPooling1D() (emb_item_desc)
    #rnn_layer2 = GRU(8, recurrent_dropout=0.0) (emb_category_name_split)
    rnn_layer2 = GlobalMaxPooling1D() (emb_category_name_split)
    
    #rnn_layer3 = GRU(8, recurrent_dropout=0.0) (emb_name)
    rnn_layer3 = GlobalMaxPooling1D() (emb_name)
    #rnn_layer4 = GRU(8, recurrent_dropout=0.0) (emb_ntk)
    rnn_layer4 = GlobalMaxPooling1D() (emb_ntk)
    
    #main layer
    main_l = concatenate([
        Flatten() (emb_brand)
        , Flatten() (emb_item_condition)
        , rnn_layer1
        , rnn_layer2
        , rnn_layer3
        , rnn_layer4
        , num_vars
    ])
    main_l = Dropout(dr)(Dense(128,activation='relu') (main_l))
    main_l = BatchNormalization()(main_l)
    main_l = Dropout(dr)(Dense(64,activation='relu') (main_l))
    main_l = BatchNormalization()(main_l)
    
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

#model = get_model()
#model.summary()
    
print('[{}] Finished DEFINING MODEL...'.format(time.time() - start_time))

printSection('Start the RNN training #1A')


dtrain   = mergetrn[:nrow_train].iloc[trnidx]
dvalid   = mergetrn[:nrow_train].iloc[validx]
del mergetrn
gc.collect()

def make_denstrn():
    # Break it up to save memory
    densetrn_part1 = mergeEmbeddings(dtrain[:70000]) # densetrain[:nrow_train][trnidx] 
    densetrn_part2 = mergeEmbeddings(dtrain[70000:]) # densetrain[:nrow_train][trnidx] 
    densetrn = np.concatenate([densetrn_part1, densetrn_part2])
    return densetrn

densetrn = make_denstrn()
denseval = mergeEmbeddings(dvalid) # densetrain[:nrow_train][validx]
gc.collect()


bags = 2
y_pred_epochs = []


from keras.layers import Dense, GlobalAveragePooling1D, Embedding
for b in range(bags):
    val_sorted_ix = np.array(map_sort(dvalid["seq_item_description"].tolist(), dvalid["seq_name_token"].tolist()))
    printSection('Start the RNN training part1 bag%s'%(b))
    epochs = 2
    batchSize = 512 * 4
    steps = (dtrain.shape[0]/batchSize+1)*epochs
    lr_init, lr_fin = 0.01, 0.008
    lr_decay  = (lr_init - lr_fin)/steps
    model = get_model()
    K.set_value(model.optimizer.lr, lr_init)
    K.set_value(model.optimizer.decay, lr_decay)
    for i in range(epochs):
        model.fit_generator(
                            trn_generator(densetrn, dtrain, dtrain.target, batchSize)
                            , epochs=1
                            , steps_per_epoch = int(np.ceil(dtrain.shape[0]*1.*0.9/batchSize))
                            , validation_data = val_generator(denseval, dvalid, dvalid.target, batchSize)
                            , validation_steps = int(np.ceil(dvalid.shape[0]*1.*0.9/batchSize))
                            , verbose=1
                            )
    
    from keras.models import load_model
    cpuStats()
    
    '''
    Training Keras model
    '''
    printSection('Start the RNN training part2 bag%s'%(b))
    lr_ls = [0.00, 0.009, 0.008]
    for c, lr in enumerate(lr_ls): # , 0.006, 0.007,
        K.set_value(model.optimizer.lr, lr)
        model.fit_generator(
                            trn_generator(densetrn, dtrain, dtrain.target, batchSize)
                            , epochs=1#,epochs
                            , max_queue_size=4
                            , steps_per_epoch = int(np.ceil(dtrain.shape[0]*1.*0.4/batchSize))
                            , validation_data = val_generator(denseval, dvalid, dvalid.target, batchSize)
                            , validation_steps = int(np.ceil(dvalid.shape[0]*1./batchSize))
                            , verbose=2
                            )
        y_pred_epochs.append(model.predict_generator(
                        tst_generator(denseval[val_sorted_ix], dvalid.iloc[val_sorted_ix], batchSize)
                        , steps = int(np.ceil(dvalid.shape[0]*1./batchSize))
                        , max_queue_size=1 
                        , verbose=2)[val_sorted_ix.argsort()])
        model.save_weights('%s/rnn_model%s.h5'%(mdir, c+(b*len(lr_ls))))
        print("Epoch %s rmsle %s"%(epochs+c+1, eval_model(dvalid.price.values, y_pred_epochs[-1])))
        cpuStats()
        
y_pred = sum(y_pred_epochs)/len(y_pred_epochs)
print("Bagged Epoch %s rmsle %s"%(epochs+c+1, eval_model(dvalid.price.values, y_pred)))
print("Bagged FM & Nnet", rmsle(dvalid.price.values, np.expm1(predsfm)*0.5 + np.expm1(y_pred[:,0])*0.5  ))

del densetrn, denseval, dtrain, dvalid
gc.collect()
cpuStats()

'''
Inference with Keras model
'''

printSection('Start the RNN prediction')
def inference_keras(mergetst):
    tst_sorted_ix = np.array(map_sort(mergetst["seq_item_description"].tolist(), mergetst["seq_name_token"].tolist()))
    densetest = mergeEmbeddings(mergetst)
    gc.collect()
    cpuStats()
    yspred_epochs = []
    for c in range(len(lr_ls)*bags):
        # Drop the stopwords
        #if c>1:
        #    mergetst = seqTokenStopWDrop(mergetst)
            
        model.load_weights('%s/rnn_model%s.h5'%(mdir, c))
        yspred_epochs.append(model.predict_generator(
                        tst_generator(densetest[tst_sorted_ix], mergetst.iloc[tst_sorted_ix], batchSize)
                        , steps = int(np.ceil(mergetst.shape[0]*1./batchSize))
                        , max_queue_size=1 
                        , verbose=2)[tst_sorted_ix.argsort()])
    cpuStats()
    yspred = sum(yspred_epochs)/len(yspred_epochs)
    return yspred

yspredls = [inference_keras(df) for df in tstls]
yspred   = np.squeeze(np.concatenate(yspredls, axis= 0 ))
yspred.shape
test_ids_df = pd.concat([df[['test_id']] for df in tstls])
    
printSection('Start Submitting....')

bag_preds = np.expm1(predsFM)*0.5 + np.expm1(yspred)*0.5  
print('[{}] Finished predicting test set...'.format(time.time() - start_time))
submission = test_ids_df.copy()
submission["price"] = bag_preds
submission.to_csv("./myBag_0602.csv", index=False)
print('[{}] Finished submission...'.format(time.time() - start_time))

submFM = test_ids_df.copy()
submRNN = test_ids_df.copy()
submFM["price"] = np.expm1(predsFM)
submRNN["price"] = np.expm1(yspred)
submFM.to_csv("./FM1002.csv", index=False)
submRNN.to_csv("./RNN1002.csv", index=False)
#submission.to_csv("./myBag"+log_subdir+"_{:.6}.csv".format(v_rmsle), index=False)
print('[{}] Finished submission...'.format(time.time() - start_time))
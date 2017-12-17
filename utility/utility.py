import pyximport; pyximport.install()
import gc, sys, os
import time
from tqdm import tqdm 
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelBinarizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

NUM_BRANDS = 4000
NUM_CATEGORIES = 1000
NAME_MIN_DF = 10
MAX_FEATURES_ITEM_DESCRIPTION = 50000

def rmsle(y, y0):
     assert len(y) == len(y0)
     return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))
    
def split_cat(text):
    try: return text.split("/")
    except: return ("No Label", "No Label", "No Label")
    
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
    

class LabelCount(object):
    '''
    https://github.com/lgmoneda/greenpyce
    '''
    def __init__(self, columns, new_column=False):
        self.count_dict = {}
        self.columns = columns
        self.new_column = new_column
    def fit(self, df):
        for column in self.columns:
            count = df[column].value_counts()
            self.count_dict[column] = count.to_dict()
    def transform(self, df):
        for column in self.columns:
            new_column_name = column
            if self.new_column:
                new_column_name = column + "_label_count"
            missing = 1
            df[new_column_name] = df[column].apply(lambda x : self.count_dict[column].get(x, missing))            
            
class StringLength(object):
    def __init__(self, columns, new_column=False):
        self.columns = columns
        self.new_column = new_column
    def transform(self, df):
        for column in self.columns:
            new_column_name = column
            if self.new_column:
                new_column_name = column + "_str_len"
            missing = 1
            df[new_column_name] = df[column].str.len().values        
            
class Glove(object):
    '''
    https://github.com/lgmoneda/greenpyce
    '''
    def __init__(self, columns, folder = '../feat', verbose = 0, scale = True, svd = True):
        self.verbose = verbose
        self.embeddings_index = {}
        self.columns = columns
        self.scale = scale
        self.svd   = svd
        self.verbose = verbose
        f = open(os.path.join(folder, 'glove.840B.300d.txt'))
        for line in tqdm(f):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float16')
            self.embeddings_index[word] = coefs
        f.close()
        if self.verbose>0: print('Found %s word vectors.' % len(self.embeddings_index))
        
        self.col_embeddings = {}
        
    def sent2vec(self, s):
        words = str(s).lower().decode('utf-8')
        words = word_tokenize(words)
        words = [w for w in words if not w in stop_words]
        #words = [w for w in words if w.isalpha()]
        M = []
        for w in words:
            try:
                M.append(self.embeddings_index[w])
            except:
                continue
        M = np.array(M)
        v = M.sum(axis=0)
        if type(v) != np.ndarray:
            return np.zeros(300)
        return v / np.sqrt((v ** 2).sum())
        
    def transform(self, df, n_components = 50):
        
        for column in self.columns:
            if self.verbose > 0 : print 'Start... ' + column
            col_strings = df[column].str.replace('/', ' ')
            unq_strings = col_strings.unique()
            if self.verbose > 0 :
                unq_strings_dict = dict(zip(unq_strings, [self.sent2vec(x) for x in tqdm(unq_strings)]))
                print 'Get embeddings ' + column
                self.col_embeddings[column] = [unq_strings_dict[x] for x in tqdm(col_strings)]
            else:
                unq_strings_dict = dict(zip(unq_strings, [sent2vec(x) for x in unq_strings]))
                self.col_embeddings[column] = [unq_strings_dict[x] for x in col_strings]
            self.col_embeddings[column] = np.array(self.col_embeddings[column])
            # scale the data before any neural net:
            if self.svd:
                if self.verbose > 0 : print 'Truncate embeddings ' + column
                if n_components<300:
                    svd = TruncatedSVD(n_components= n_components)
                    self.col_embeddings[column] = svd.fit_transform(self.col_embeddings[column])
            if self.scale:
                if self.verbose > 0 : print 'Scale embeddings ' + column
                scl = StandardScaler()
                self.col_embeddings[column] = scl.fit_transform(self.col_embeddings[column]).astype('float16')
        return np.hstack(self.col_embeddings.values()).astype('float16')

class TargetEncoder(object):
    '''
    https://github.com/lgmoneda/greenpyce
    '''
    def __init__(self, columns, target, new_column=False):
        self.means_dict = {}
        self.columns = columns
        self.target = target
        self.new_column = new_column            
    def fit(self, df):
        for column in self.columns:
            group = pd.groupby(df[[column, self.target]], column).mean()
            self.means_dict[column] = group.to_dict()            
    def transform(self, df):
        for column in self.columns:            
            new_column_name = column
            if self.new_column:
                new_column_name = column + "_target_encoding"            
            missing = np.mean(np.array(self.means_dict[column][self.target].values()))
            df[new_column_name] = df[column].apply(lambda x : self.means_dict[column][self.target].get(x, missing))


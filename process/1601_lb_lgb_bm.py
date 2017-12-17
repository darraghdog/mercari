#changes:
# https://www.kaggle.com/iamprateek/submission-to-mercari-price-suggestion-challenge
#modified main method to remove zero priced product
#required libraries
import pyximport; pyximport.install()
import gc, sys, os
sys.path.append('/home/darragh/mercari')
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from utility import utility

os.chdir('/home/darragh/mercari/data')

NUM_BRANDS = 4000
NUM_CATEGORIES = 1000
NAME_MIN_DF = 10
MAX_FEATURES_ITEM_DESCRIPTION = 50000        
            
def main():
    
    start_time = time.time()
    train = pd.read_table('../data/train.tsv', engine='c')
    test = pd.read_table('../data/test.tsv', engine='c')
    
    # Make a stratified split
    n_fold, n_samples = 5, train.shape[0] 
    np.random.seed(0)
    split_idx_ = train['shipping']*100000 + train['item_condition_id'] * 1000 + \
                        train['price'] + np.random.randint(0,50,size=(train.shape[0] ))
    split_idx_ = split_idx_.sort_values()
    folds = [split_idx_.index.values[i::5] for i in range(n_fold)]
    idx = np.zeros((train.shape[0]), np.int)
    for i, f in zip(range(n_fold), folds):
        idx[f]=i
    
    
    print('[{}] Finished to load training and test data'.format(time.time() - start_time))
    print('Train shape: ', train.shape)
    print('Test shape: ', test.shape)
    nrow_test = train.shape[0]
    dftt = train[(train.price < 1.0)]
    train = train.drop(train[(train.price < 1.0)].index)
    del dftt['price']
    nrow_train = train.shape[0]
    y = np.log1p(train["price"])
    merge = pd.concat([train, dftt, test])
    submission = test[['test_id']]
    
    del train
    del test
    gc.collect()
    
    merge['general_cat'], merge['subcat_1'], merge['subcat_2'] = \
    zip(*merge['category_name'].apply(lambda x: utility.split_cat(x)))
        
    print('[{}] Split categories completed.'.format(time.time() - start_time))
    strcols =  ["brand_name", "category_name", "general_cat", "subcat_1", "subcat_2"]
    lc = utility.LabelCount(strcols, new_column=True)
    lc.fit(merge)
    lc.transform(merge)
    sl = utility.StringLength(strcols, new_column=True)
    sl.transform(merge)  
    del lc, sl; gc.collect()
    print('[{}] Cat counter completed.'.format(time.time() - start_time))
    '''
    # Get the Glove embeddings 
    glove = utility.Glove(["name", "category_name"], verbose = 1)
    X_embeddings = glove.transform(merge)  
    '''
    del glove
    gc.collect()
    print('[{}] Add embeddings.'.format(time.time() - start_time))
    
    merge.drop('category_name', axis=1, inplace=True)
    print('[{}] Add counts and lengths.'.format(time.time() - start_time))

    utility.handle_missing_inplace(merge)
    utility.cutting(merge)
    utility.to_categorical(merge)

    cv = CountVectorizer(min_df=NAME_MIN_DF)
    X_name = cv.fit_transform(merge['name']).astype('float16')
    print('[{}] Count vectorize `name` completed.'.format(time.time() - start_time))

    cv = CountVectorizer()
    X_category1 = cv.fit_transform(merge['general_cat']).astype('float16')
    X_category2 = cv.fit_transform(merge['subcat_1']).astype('float16')
    X_category3 = cv.fit_transform(merge['subcat_2']).astype('float16')
    print('[{}] Count vectorize `categories` completed.'.format(time.time() - start_time))

    tv = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION,
                         ngram_range=(1, 3),
                         stop_words='english')
    X_description = tv.fit_transform(merge['item_description']).astype('float16')
    print('[{}] TFIDF vectorize `item_description` completed.'.format(time.time() - start_time))

    lb = LabelBinarizer(sparse_output=True)
    X_brand = lb.fit_transform(merge['brand_name']).astype('float16')
    print('[{}] Label binarize `brand_name` completed.'.format(time.time() - start_time))

    X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']],
                                          sparse=True).values).astype('float16')
    print('[{}] Get dummies on `item_condition_id` and `shipping` completed.'.format(time.time() - start_time))

    base_cols = [col for col in merge.columns if any(x in col for x in ['_label_count', '_str_len'])]
    sparse_merge = hstack((csr_matrix(merge[base_cols].fillna(-1).values), \
                           X_dummies, X_description, X_brand, X_category1, \
                           X_category2, X_category3, X_name)).tocsr().astype('float16')
    del X_dummies, X_description, X_brand, X_category1
    del X_category2, X_category3, X_name
    del merge
    '''
    X_embeddings = csr_matrix(X_embeddings)
    sparse_merge = hstack((sparse_merge, X_embeddings)).tocsr()
    del X_embeddings
    '''    
    print('[{}] Create sparse merge completed'.format(time.time() - start_time))

    X = sparse_merge[:nrow_train]
    X_test = sparse_merge[nrow_test:]
    
    model = Ridge(alpha=.5, copy_X=True, fit_intercept=True, max_iter=100,
      normalize=False, random_state=101, solver='auto', tol=0.01)
    model.fit(X, y)
    print('[{}] Train ridge completed'.format(time.time() - start_time))
    predsR = model.predict(X=X_test)
    print('[{}] Predict ridge completed'.format(time.time() - start_time))

    train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size = 0.1, random_state = 144) 
    d_train = lgb.Dataset(train_X, label=train_y, max_bin=8192)
    d_valid = lgb.Dataset(valid_X, label=valid_y, max_bin=8192)
    watchlist = [d_train, d_valid]
    
    params = {
        'learning_rate': 0.65,
        'application': 'regression',
        'max_depth': 4, #3,
        'num_leaves': 60,
        'verbosity': -1,
        'metric': 'RMSE',
        'data_random_seed': 1,
        'bagging_fraction': 0.5,
        'nthread': 12
    }

    params2 = {
        'learning_rate': 0.85,
        'application': 'regression',
        'max_depth': 4, #3,
        'num_leaves': 140,
        'verbosity': -1,
        'metric': 'RMSE',
        'data_random_seed': 2,
        'bagging_fraction': 1,
        'nthread': 12
    }

    model = lgb.train(params, train_set=d_train, num_boost_round=8500, valid_sets=watchlist, \
    early_stopping_rounds=1000, verbose_eval=10) 
    # [8500]  training's rmse: 0.39889        valid_1's rmse: 0.445258 (baseline)
    # [8500]  training's rmse: 0.395031       valid_1's rmse: 0.445404 (with counts)
    # [3968]  training's rmse: 0.382547       valid_1's rmse: 0.451722 (with counts, embedding and depth 4)
    predsL = model.predict(X_test)
    
    print('[{}] Predict lgb 1 completed.'.format(time.time() - start_time))
    
    train_X2, valid_X2, train_y2, valid_y2 = train_test_split(X, y, test_size = 0.1, random_state = 101) 
    d_train2 = lgb.Dataset(train_X2, label=train_y2, max_bin=8192)
    d_valid2 = lgb.Dataset(valid_X2, label=valid_y2, max_bin=8192)
    watchlist2 = [d_train2, d_valid2]

    model = lgb.train(params2, train_set=d_train2, num_boost_round=5000, valid_sets=watchlist2, \
    early_stopping_rounds=500, verbose_eval=50) 
    predsL2 = model.predict(X_test)

    print('[{}] Predict lgb 2 completed.'.format(time.time() - start_time))

    preds = predsR*0.3 + predsL*0.35 + predsL2*0.35

    submission['price'] = np.expm1(preds)
    submission.to_csv("submission_ridge_2xlgbm_seven.csv", index=False)

if __name__ == '__main__':
    main()

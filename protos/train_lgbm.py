import os
import sys
import pandas as pd
import numpy as np
from statistics import mode
from collections import Counter
from scipy import sparse as ssp
from tqdm import tqdm
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import KFold, StratifiedKFold, ParameterGrid, train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, roc_curve, auc, mean_squared_error
from sklearn.utils import shuffle
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import cross_validation
import xgboost as xgb
import lightgbm as lgb
import datetime
import nltk
nltk.download('stopwords')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from load_data import load_train_data, load_test_data

args = sys.argv
if len(args) == 1:
        DIR = os.path.join('result', 'tmp')
elif len(args) == 2:
        DIR = os.path.join('result', args[1] + '.' + datetime.datetime.today().strftime('%Y%m%d'))
else:
        print ('too many arguments')
        exit   
        
if not os.path.exists(DIR):
        os.mkdir(DIR)

logger = getLogger(__name__)
log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')

handler = StreamHandler()
handler.setLevel('INFO')
handler.setFormatter(log_fmt)
logger.addHandler(handler)

handler = FileHandler(os.path.join(DIR, 'train.py.log'), 'a')
handler.setLevel(DEBUG)
handler.setFormatter(log_fmt)
logger.setLevel(DEBUG)
logger.addHandler(handler)

def calc_rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred)).astype('float32')

def fill_null(df):
        '''
        im1_mode = mode(df['image_top_1'])     
        fill_val = {'price': df['price'].mean(), 'image_top_1': im1_mode}
        new_df = df.fillna(fill_val)
        return new_df
        '''
        new_df = df.fillna({'price': np.mean(df['price']), 'description': ' '})
        return new_df

def calc_tfidf(df):
        n_feature = 100
        stop_words = nltk.corpus.stopwords.words('russian')        
        vec = TfidfVectorizer(max_features=n_feature, stop_words=stop_words)
        
        desc = vec.fit_transform(df['description'])
        desc = np.array(desc.todense(), dtype=np.float32)
         
        pca = PCA(n_components=22)
        pca.fit(desc)
        trans_desc = pca.fit_transform(desc)        
        for i in range(22):
                df['description_' + str(i)] = trans_desc[:, i]

        title = vec.fit_transform(df['title'])        
        title = np.array(title.todense(), dtype=np.float32)
        pca.fit(title)
        trans_title = pca.fit_transform(title)
        for i in range(19):
                df['title_' + str(i)] = trans_title[:, i]

        param_combined = vec.fit_transform(df['param_combined'])
        param_combined = np.array(param_combined.todense(), dtype=np.float32)
        pca.fit(param_combined)
        trans_combined = pca.fit_transform(param_combined)
        for i in range(19):
                df['param_combined_' + str(i)] = trans_combined[:, i]
                
        return df

def likelihood_encoding(df):
        cat_features = ['user_id', 'region', 'city', 'parent_category_name', 'category_name', 'user_type']
        cv = KFold(n_splits=5, shuffle=True, random_state=0)
        for c in cat_features:
                train_enc = None
                for train_idx, valid_idx in cv.split(df):
                        enc_map = df.iloc[train_idx].groupby([c])['deal_probability'].mean()                            
                        tmp = df[c].iloc[valid_idx].map(enc_map)
                        if train_enc is None:
                                train_enc = tmp
                        else:
                                train_enc = pd.concat([train_enc, tmp])                                
                df[c] = train_enc.sort_index().values
        return df
        
        
def preprocessing(train, test):
        X_train = train.drop(['deal_probability'], axis=1)
        y_train = train['deal_probability']
 
        logger.debug('start fill missing value')
        X_train = fill_null(X_train)
        X_test = fill_null(test)

        logger.debug('start date features')
        X_train['weekday'] = X_train.activation_date.dt.weekday
        X_train['month'] = X_train.activation_date.dt.month
        X_train['day'] = X_train.activation_date.dt.day
        X_train['week'] = X_train.activation_date.dt.week
        X_test['weekday'] = X_test.activation_date.dt.weekday
        X_test['month'] = X_test.activation_date.dt.month
        X_test['day'] = X_test.activation_date.dt.day
        X_test['week'] = X_test.activation_date.dt.week
        
        logger.debug('start likelihood encoding to categorical feature')        
        train_all = pd.concat([X_train, y_train], axis=1)
        train_all_enc = likelihood_encoding(train_all)
        X_train = train_all_enc.drop(['deal_probability'], axis=1)
        y_train = train_all_enc['deal_probability']
        
        '''
        logger.debug('start label encoding to categorical feature')        
        for c in cat_features:
                le = LabelEncoder()
                le.fit(np.concatenate([X_train[c].values, X_test[c].values]))
                X_train[c] = le.transform(X_train[c].values)
                X_test[c] = le.transform(X_test[c].values)
        '''
        
        logger.debug('start sentence length')
        X_train['desc_len'] = X_train.description.apply(lambda x: len(x.split()))
        X_train['title_len'] = X_train.title.apply(lambda x: len(x.split()))
        X_train['desc_char'] = X_train.description.apply(len)
        X_train['title_char'] = X_train.title.apply(len)
        
        X_test['desc_len'] = X_test.description.apply(lambda x: len(x.split()))
        X_test['title_len'] = X_test.title.apply(lambda x: len(x.split()))
        X_test['desc_char'] = X_test.description.apply(len)
        X_test['title_char'] = X_test.title.apply(len)
        
        X_train['param_combined'] = X_train.apply(lambda row: ' '.join([str(row['param_1']), str(row['param_2']), str(row['param_3'])]), axis=1)
        X_train['param_combined'] = X_train['param_combined'].fillna(' ')
        X_train['param_combined_len'] = X_train['param_combined'].apply(lambda x : len(x.split()))        
        X_test['param_combined'] = X_test.apply(lambda row: ' '.join([str(row['param_1']), str(row['param_2']), str(row['param_3'])]), axis=1)
        X_test['param_combined'] = X_test['param_combined'].fillna(' ')
        X_test['param_combined_len'] = X_test['param_combined'].apply(lambda x : len(x.split()))        
        
        logger.debug('start tf-idf')
        tfidf_train = calc_tfidf(X_train)
        tfidf_test = calc_tfidf(X_test)

        X_train.describe()
        logger.debug('start log price and item_seq_number' )
        X_train['log_price'] = np.log(X_train.price)
        X_train['log_item_seq_number'] = np.log(X_train.item_seq_number)
        X_test['log_price'] = np.log1p(X_test.price)
        X_test['log_item_seq_number'] = np.log1p(X_test.item_seq_number)        
        
        drop_cols = ['item_id', 'activation_date', 'description', 'title', 'image', 'param_1', 'param_2', 'param_3', 'param_combined', 'price']
        X_train = X_train.drop(drop_cols, axis=1)
        X_test = X_test.drop(drop_cols, axis=1)
        return X_train, y_train, X_test

if __name__=='__main__':
        TRAIN_DATA = '../input/train.csv'
        TEST_DATA = '../input/test.csv'
        SAMPLE_SUBMIT_FILE = '../input/sample_submission.csv'
        '''        
        TRAIN_DATA = '../input/train.short.csv'
        TEST_DATA = '../input/test.short.csv'
        SAMPLE_SUBMIT_FILE = '../input/sample_submission.short.csv'
        '''
        TRAIN_TYPE = 'cv'

        logger.debug('start load data')        
        train_org = pd.read_csv(TRAIN_DATA, parse_dates=['activation_date'])
        test_org = pd.read_csv(TEST_DATA, parse_dates=['activation_date'])

        X_train, y_train, X_test = preprocessing(train_org, test_org)

        use_cols = X_train.columns.values        
        
        '''
        params =  {
                'num_leaves': [31, 63, 128],
                'max_depth': [-1],
                #'min_data_in_leaf': [100, 500, 1000],
                'min_data_in_leaf': [10],
                'n_estimators': [10000]                
        }
        '''
        
        params = {
                'num_leaves': 31,
                'max_depth': -1,
                #'min_data_in_leaf': 10,
                'min_data_in_leaf': 500,
                'n_estimators': 2201
        }
     
        cv = KFold(n_splits=5, shuffle=True, random_state=0)
        early_stopping_rounds=100
        if TRAIN_TYPE is 'grid':
                i = 1
                min_score = 10000
                min_params = None
                params_grid = list(ParameterGrid(params))
                for params in tqdm(params_grid):
                        logger.debug('grid {}/{}, {}'.format(i, len(params_grid), params))
                        
                        list_scores = []
                        list_best_iterations = []
                        for train_idx, valid_idx in cv.split(X_train, y_train):
                                _X_train = X_train.iloc[train_idx]
                                _y_train = y_train.iloc[train_idx]
                                _X_valid = X_train.iloc[valid_idx]
                                _y_valid = y_train.iloc[valid_idx]
                                
                                clf = lgb.LGBMRegressor(boosting_type= 'gbdt', objective = 'regression', **params)
                                clf.fit(_X_train, _y_train, eval_set=[(_X_valid, _y_valid)], early_stopping_rounds=early_stopping_rounds, eval_metric='rmse')                                                        
                                pred_valid = clf.predict(_X_valid, num_iteration=clf.best_iteration_)
                                list_scores.append(calc_rmse(_y_valid, pred_valid))
                                list_best_iterations.append(clf.best_iteration_)
                        
                        score = np.mean(list_scores)
                        if min_score > score:
                                min_score = score
                                params['n_estimators'] = int(np.mean(list_best_iterations))
                                min_params = params
                        logger.debug('current rmse: {}, min rmse: {}'.format(score, min_score))
                        i += 1
                logger.debug('end grid search: {}, {}'.format(min_score, min_params))
                
        elif TRAIN_TYPE is 'cv':
                logger.debug('start cross validation')

                list_scores = []
                for train_idx, valid_idx in cv.split(X_train, y_train):
                        _X_train = X_train.iloc[train_idx]
                        _y_train = y_train.iloc[train_idx]
                        _X_valid = X_train.iloc[valid_idx]
                        _y_valid = y_train.iloc[valid_idx]
                        
                        clf = lgb.LGBMRegressor(boosting_type= 'gbdt', objective = 'regression', **params)
                        clf.fit(_X_train, _y_train, eval_metric='rmse')
                        pred_valid = clf.predict(_X_valid)
                        list_scores.append(calc_rmse(_y_valid, pred_valid))
                logger.debug('all rmse: {}'.format(list_scores))
                logger.debug('mean rmse: {}'.format(np.mean(list_scores)))

        elif TRAIN_TYPE is 'simple':
                pass
                     
        logger.debug('start predict test data')
        clf = lgb.LGBMRegressor(boosting_type= 'gbdt', objective = 'regression', **params)
        clf.fit(X_train, y_train, eval_metric='rmse')
        pred_test = clf.predict(X_test)

        # Feature Importance Plot
        f, ax = plt.subplots(figsize=[7,10])
        lgb.plot_importance(clf, max_num_features=50, ax=ax)
        plt.title("Light GBM Feature Importance")
        plt.savefig(os.path.join(DIR, 'feature_importance.png'))
                
        df_submit = pd.read_csv(SAMPLE_SUBMIT_FILE)
        df_submit['deal_probability'] = pred_test
        df_submit['deal_probability'].clip(0.0, 1.0, inplace=True)
        df_submit.to_csv(os.path.join(DIR, 'submission.csv'), index=False)
        logger.info('end')

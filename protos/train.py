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
from sklearn.model_selection import KFold, StratifiedKFold, ParameterGrid
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, roc_curve, auc, mean_squared_error
from sklearn.utils import shuffle
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
import nltk
nltk.download('stopwords')
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

SAMPLE_SUBMIT_FILE = '../input/sample_submission.csv'

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
        return np.sqrt(mean_squared_error(y_true, y_pred))       

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
        print(pca.explained_variance_ratio_)
        print(np.cumsum(pca.explained_variance_ratio_))
        for i in range(19):
                df['title_' + str(i)] = trans_title[:, i]               
        return df
                
if __name__=='__main__':
        logger.debug('start load train data')
        df_train = load_train_data()
        #df_train = df_train.iloc[:10000,:]
        X_train = df_train.drop(['deal_probability'], axis=1)
        y_train = df_train['deal_probability']

        logger.debug('start load test data')        
        X_test = load_test_data()
        
        logger.debug('start fill null')
        X_train = fill_null(X_train)
        X_test = fill_null(X_test)

        X_train["Weekday"] = X_train['activation_date'].dt.weekday
        X_train["Weekd of Year"] = X_train['activation_date'].dt.week
        X_train["Day of Month"] = X_train['activation_date'].dt.day
        X_test["Weekday"] = X_test['activation_date'].dt.weekday
        X_test["Weekd of Year"] = X_test['activation_date'].dt.week
        X_test["Day of Month"] = X_test['activation_date'].dt.day        
        
        logger.debug('start cat_features')
        cat_features = ['user_id', 'region', 'city', 'parent_category_name', 'category_name', 'user_type']
        for c in cat_features:
                le = LabelEncoder()
                logger.debug('start: {}'.format(c))
                le.fit(np.concatenate([X_train[c].values, X_test[c].values]))
                X_train[c] = le.transform(X_train[c].values)
                X_test[c] = le.transform(X_test[c].values)

        logger.debug('start tf-idf')
        tfidf_train = calc_tfidf(X_train)
        tfidf_test = calc_tfidf(X_test)

        drop_cols = ['item_id', 'activation_date', 'description', 'title', 'image', 'param_1', 'param_2', 'param_3']
        X_train = X_train.drop(drop_cols, axis=1)
        X_test = X_test.drop(drop_cols, axis=1)
        use_cols = X_train.columns.values
        X_train = X_train.values
        X_test = X_test.values
        y_train = y_train.values
                
        '''
        all_params = {'max_depth': [3, 5, 7],
                      'learning_rate': [0.1],
                      'min_child_weight': [3, 5, 10],
                      'n_estimators': [10000],
                      'colsample_bytree': [0.8, 0.9],
                      'colsample_bylevel': [0.8],
                      'reg_alpha': [0, 0.1],
                      'max_delta_step': [0.1],
                      'seed': [0]}
        '''
         
        all_params = {'max_depth': [5, 7, 10],
                      'learning_rate': [0.1],
                      'min_child_weight': [5],
                      'n_estimators': [10000],
                      'colsample_bytree': [0.8],
                      'colsample_bylevel': [0.8],
                      'reg_alpha': [0],
                      'max_delta_step': [0.1],
                      'seed': [0]}

        logger.debug('all params: {}'.format(all_params))

        cv = KFold(n_splits=5, shuffle=True, random_state=0)
        
        min_rmse = 100
        min_params = None
        grid_cnt = 1
        logger.debug('start grid search')
        for params in tqdm(list(ParameterGrid(all_params))):
                logger.debug('{}/{}'.format(grid_cnt, len(list(ParameterGrid(all_params)))))
                logger.debug('params: {}'.format(params))
                                
                list_rmse = []
                list_best_iterations = []
                for train_idx, valid_idx in cv.split(X_train, y_train):
                        _X_train = X_train[train_idx]
                        _y_train = y_train[train_idx]                                
                        _X_valid = X_train[valid_idx]
                        _y_valid = y_train[valid_idx]

                        clf = xgb.sklearn.XGBRegressor(**params)
                        clf.fit(_X_train, _y_train, eval_set=[(_X_valid, _y_valid)], early_stopping_rounds=20, eval_metric='rmse')                        
                        pred = clf.predict(_X_valid, ntree_limit=clf.best_ntree_limit)
                        rmse = calc_rmse(_y_valid, pred)
                        
                        list_rmse.append(rmse)
                        list_best_iterations.append(clf.best_iteration)
                        break
                
                params['n_estimators'] = int(np.mean(list_best_iterations))
                mean_rmse = np.mean(list_rmse)
                if min_rmse > mean_rmse:
                        min_rmse = mean_rmse
                        min_params = params
                logger.debug('rmse: {}'.format(rmse))
                grid_cnt += 1

        logger.debug('minimum params: {}'.format(min_params))
        logger.debug('minimum rmse: {}'.format(min_rmse))
        logger.debug('end grid search')
        
        f, ax = plt.subplots(figsize=[7,10])
        lgb.plot_importance(lgb_clf, max_num_features=50, ax=ax)
        plt.title("Light GBM Feature Importance")
        plt.savefig('feature_import.png')
        
        logger.debug('predict test data start')
        clf = xgb.sklearn.XGBRegressor(**min_params)
        clf.fit(X_train, y_train)
        pred_test = clf.predict(X_test)
        logger.debug('predict test data end')

        df_submit = pd.read_csv(SAMPLE_SUBMIT_FILE)
        df_submit['deal_probability'] = pred_test
        df_submit['deal_probability'].clip(0.0, 1.0, inplace=True)

        df_submit.to_csv(os.path.join(DIR, 'submission.csv'), index=False)
        logger.info('end')

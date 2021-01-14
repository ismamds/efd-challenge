import os
import pandas as pd
import sqlite3
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit

problem_title = 'Prediction of European Soccer League match results'
_target_column_name = 'Result'
_ignore_column_names = []
_prediction_label_names = ['Home_win', 'Draw', 'Away_win']

# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)

# An object implementing the workflow
workflow = rw.workflows.Estimator()

score_types = [
    rw.score_types.Accuracy(name='acc', precision=4),
]


def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=8, test_size=0.2, random_state=57)
    return cv.split(X, y)


# READ DATA

def _read_data(path, train_test):
    
    conn = sqlite3.connect(os.path.join(path, 'data', 'matches.csv'))
    
    query = "SELECT * FROM Match;"

    df = pd.read_sql_query(query,conn)

    df = df.drop(df.columns[range(77,85)],axis='columns')

    df = df.set_index('id')
    
    X = df.drop(['home_team_goal',
                 'away_team_goal',
                 'season','date',
                 'country_id',
                 'league_id',
                 'match_api_id'],axis='columns')
    
    Y = pd.Series(0,index=df.index)

    away = df[df['home_team_goal']<df['away_team_goal']].index
    draw = df[df['home_team_goal']==df['away_team_goal']].index

    Y[away] = 1
    Y[draw] = 2
    
    if train_test == 'train' :
        train_index = df[df['season'].isin(['2008/2009','2009/2010',
                                            '2010/2011','2011/2012',
                                            '2012/2013','2013/2014'])].index
        X_train = X.loc[train_index.values].set_index(pd.Index(range(1,len(train_index))))
        y_train = Y[train_index].set_axis(pd.Index(range(1,len(train_index))))
        return X_train, y_train
    
    if train_test == 'test' :
        test_index = df[df['season'].isin(['2014/2015','2015/2016'])].index
        X_test = X.loc[test_index.values].set_index(pd.Index(range(1,len(test_index))))
        y_test = Y[test_index].set_axis(pd.Index(range(1,len(test_index))))
        return X_test, y_test


def get_train_data(path='.'):
    return _read_data(path, 'train')


def get_test_data(path='.'):
    return _read_data(path, 'test')
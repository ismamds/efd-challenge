import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit
import zipfile

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
    
    if os.path.isfile('matches.csv'):
        df = pd.read_csv('matches.csv')
    
    else:
        path_to_zip_file = os.path.join(path, 'data', 'matches.zip')
        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall(path)
        df = pd.read_csv('matches.csv')

    df = df.set_index('id')
    
    X = df.drop(['home_team_goal',
                 'away_team_goal',
                 'goal',
                 'shoton', 'shotoff', 'foulcommit', 'card', 'cross', 'corner',
                 'Unnamed: 0',
                 'possession',
                 'season',
                 'country_id',
                 'league_id',
                 'match_api_id'] + \
                #[f"away_player_Y{i}" for i in range(1, 12)] + \
                [f"away_player_{i}" for i in range(1, 12)] + \
                [f"home_player_{i}" for i in range(1, 12)] 
                #[f"away_player_X{i}" for i in range(1, 12)] + \
                #[f"home_player_Y{i}" for i in range(1, 12)] + \
                #[f"home_player_X{i}" for i in range(1, 12)
                ,
                axis='columns')
    
    Y = pd.Series(0,index=df.index)

    away = df[df['home_team_goal']<df['away_team_goal']].index
    draw = df[df['home_team_goal']==df['away_team_goal']].index

    Y[away] = 1
    Y[draw] = 2
    
    if train_test == 'train' :
        train_index = df[df['season'].isin(['2008/2009','2009/2010',
                                            '2010/2011','2011/2012',
                                            '2012/2013','2013/2014'])].index
        X_train = X.loc[train_index.values].set_index(pd.Index(range(len(train_index))))
        y_train = Y[train_index].set_axis(pd.Index(range(len(train_index))))
        return X_train, y_train
    
    if train_test == 'test' :
        test_index = df[df['season'].isin(['2014/2015','2015/2016'])].index
        X_test = X.loc[test_index.values].set_index(pd.Index(range(len(test_index))))
        y_test = Y[test_index].set_axis(pd.Index(range(len(test_index))))
        return X_test, y_test


def get_train_data(path='.'):
    return _read_data(path, 'train')


def get_test_data(path='.'):
    return _read_data(path, 'test')
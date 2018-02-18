import pandas as pd
import matplotlib as plt
import numpy as np

df = pd.read_csv('train.csv')

#list(df.loc[df.isFraud == 1].type.drop_duplicates().values)

X = df.loc[(df.type == 'TRANSFER') | (df.type == 'CASH_OUT')]
X = X.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis = 1)

y = X['isFraud']
X.drop('isFraud',axis=1, inplace=True)

X.loc[X.type == 'TRANSFER', 'type'] = 0
X.loc[X.type == 'CASH_OUT', 'type'] = 1
temp = X.oldbalanceDest
X.loc[(X.oldbalanceDest == 0) & (X.newbalanceDest == 0) & (X.amount != 0), ['oldbalanceDest', 'newbalanceDest']] = - 1
X.loc[(X.oldbalanceOrig == 0) & (X.newbalanceOrig == 0) & (X.amount != 0), ['oldbalanceOrig', 'newbalanceOrig']] = np.nan

X['errorbalanceOrig'] = X.newbalanceOrig + X.amount - X.oldbalanceOrig
X['errorbalanceDest'] = X.oldbalanceDest + X.amount - X.newbalanceDest

from sklearn.cross_validation import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 0)

import xgboost as xg
weights = (y == 0).sum() / (1.0 * (y == 1).sum())
xgb = xg.XGBClassifier(max_depth = 3, scale_pos_weight = weights, n_jobs = 4)

from sklearn.metrics import average_precision_score
y_pred = xgb.fit(X_train, y_train).predict_proba(X_val)
average_precision_score(y_val, y_pred[:, 1])

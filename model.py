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
import numpy as np
import pandas as  pd
from sklearn import linear_model

data_test = pd.read_csv("F:/python/kaggle/Tatanic/dataset/test.csv")
data_train = pd.read_csv("F:/python/kaggle/Tatanic/dataset/train.csv")


data_train.loc[data_train.Cabin.notnull(), 'Cabin'] = 1
data_train.loc[data_train.Cabin.isnull(), 'Cabin'] = 0

data_train.loc[data_train.Age.isnull(), 'Age'] = data_train['Age'].mean()
data_train = data_train.drop(['Name', 'Ticket','Pclass', 'Embarked', 'Sex'], axis = 1)

dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix = 'Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix = 'Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix = 'Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix = 'Pclass')

data_train = data_train.drop(['Name', 'Ticket','Pclass', 'Embarked', 'Sex'], axis = 1)

data_train = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis =1)


scaler = prepro.StandardScaler().fit(data_train['Age'].reshape(-1, 1))
data_train['Age'] = scaler.transform(data_train['Age'].reshape(-1, 1))
scaler = prepro.StandardScaler().fit(data_train['Fare'].reshape(-1,1))
data_train['Fare'] = scaler.transform(data_train['Fare'].reshape(-1, 1))

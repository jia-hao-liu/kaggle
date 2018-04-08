import numpy as np
import pandas as  pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import sklearn.preprocessing as prepro
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor)
from sklearn.model_selection import learning_curve



data_test = pd.read_csv("F:/python/kaggle/Tatanic/dataset/test.csv")
data_train = pd.read_csv("F:/python/kaggle/Tatanic/dataset/train.csv")



data_train.loc[data_train.Cabin.notnull(), 'Cabin'] = 1
data_train.loc[data_train.Cabin.isnull(), 'Cabin'] = 0
data_test.loc[data_test.Cabin.notnull(), 'Cabin'] = 1
data_test.loc[data_test.Cabin.isnull(), 'Cabin'] = 0

data_train.loc[data_train.Age.isnull(), 'Age'] = data_train['Age'].mean()
data_test.loc[data_test.Age.isnull(), 'Age'] = data_test['Age'].mean()


data_test.loc[data_test.Fare.isnull(), 'Fare'] = data_test['Fare'].mean()

scaler = prepro.StandardScaler().fit(data_train['Age'].reshape(-1, 1))
data_train['Age'] = scaler.transform(data_train['Age'].reshape(-1, 1))
scaler = prepro.StandardScaler().fit(data_train['Fare'].reshape(-1,1))
data_train['Fare'] = scaler.transform(data_train['Fare'].reshape(-1, 1))

scaler = prepro.StandardScaler().fit(data_test['Age'].reshape(-1, 1))
data_test['Age'] = scaler.transform(data_test['Age'].reshape(-1, 1))
scaler = prepro.StandardScaler().fit(data_test['Fare'].reshape(-1,1))
data_test['Fare'] = scaler.transform(data_test['Fare'].reshape(-1, 1))


dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix = 'Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')


dummies_Sex = pd.get_dummies(data_train['Sex'], prefix = 'Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix = 'Pclass')



data_train = data_train.drop(['Name', 'Ticket','Pclass', 'Embarked', 'Sex', 'Cabin'], axis = 1)

data_train = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis =1)


dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix = 'Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')

dummies_Sex = pd.get_dummies(data_test['Sex'], prefix = 'Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix = 'Pclass')


data_test = data_test.drop(['Name', 'Ticket','Pclass', 'Embarked', 'Sex', 'Cabin'], axis = 1)

data_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis =1)



# 用正则取出我们要的属性值
train_df = data_train.filter(regex='Survived|Age|SibSp|Parch|Fare|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.as_matrix()

test_df = data_test.filter(regex='Survived|Age|SibSp|Parch|Fare|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
test_np = test_df.as_matrix()

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

# LogisticRegression
clf_l = linear_model.LogisticRegression(C = 0.3, penalty = 'l1', tol = 1e-6)


clf_l.fit(X,y)



#y_test = bagging_clf.predict(test_np)

y_test_l = clf_l.predict(test_np)


#RandomForestRegressor
clf_c = RandomForestClassifier(max_depth = 5, n_estimators = 46)
clf_c.fit(X,y)
y_test_c = clf_c.predict(test_np)


result = pd.DataFrame({'PassengerId': data_test['PassengerId'], 'Survived': y_test_c.astype(np.int32)})


result.to_csv("F:\python\kaggle\Tatanic/dataset\\resultc.csv", index = False)

result = pd.DataFrame({'PassengerId': data_test['PassengerId'], 'Survived': y_test_l.astype(np.int32)})


result.to_csv("F:\python\kaggle\Tatanic/dataset\\resultl.csv", index = False)

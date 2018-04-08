import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn import linear_model
import sklearn.preprocessing as prepro
from sklearn.ensemble import BaggingRegressor


#读取数据
data_test = pd.read_csv("F:/python/kaggle/Tatanic/dataset/test.csv")
data_train = pd.read_csv("F:/python/kaggle/Tatanic/dataset/train.csv")

data_train_2 = data_train.drop(['Name', 'Cabin'], axis = 1)

fig = plt.figure('1')


plt.subplot2grid((2,3), (0,0))
data_train.Survived.value_counts().plot(kind = 'bar')
plt.title('1 denote Survived')
plt.ylabel('Population')

plt.subplot2grid((2,3),(0,1))
data_train.Pclass.value_counts().plot(kind = 'bar')
plt.title('Pclass of passengers')
plt.ylabel('Population')

plt.subplot2grid((2,3),(0,2))
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel('Age')
plt.title('1 denote Survived')


plt.subplot2grid((2,3),(1,0), colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind = 'kde')
data_train.Age[data_train.Pclass == 2].plot(kind = 'kde')
data_train.Age[data_train.Pclass == 3].plot(kind = 'kde')
plt.xlabel(u"年龄")# plots an axis lable
plt.ylabel(u"密度")
plt.title(u"各等级的乘客年龄分布")
plt.legend((u'头等舱', u'2等舱',u'3等舱'),loc='best')



plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind = 'bar')
plt.ylabel('Population')
plt.title('The population of different Embarked')





Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df =  DataFrame({'Survived': Survived_1, 'unSurvived': Survived_0})

df.plot(kind = 'bar', stacked = True)


Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
df = DataFrame({'Survived':Survived_1, 'unSurvived': Survived_0})
df.plot(kind = 'bar', stacked = True)
plt.show()
#plt.show()







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

# fit到RandomForestRegressor之中
clf = linear_model.LogisticRegression(C = 1.0, penalty = 'l1', tol = 1e-6)
bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)
bagging_clf.fit(X, y)

y_test = bagging_clf.predict(test_np)


result = pd.DataFrame({'PassengerId': data_test['PassengerId'], 'Survived': y_test.astype(np.int32)})


result.to_csv("F:\python\kaggle\Tatanic/dataset\\result1.csv", index = False)

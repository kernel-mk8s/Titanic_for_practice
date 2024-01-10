# ライブラリのインポート
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.neighbors import  KNeighborsClassifier

import warnings
warnings.filterwarnings('ignore')


# データの読み込み
path = r""

df = pd.read_csv(path + r"\train.csv")
df_test = pd.read_csv(path + r"\test.csv")


# データの補完、特徴量の作成
# for Age
age_male = df[df['Sex']=='male']['Age'].mean()
age_female =df[(df['Sex']=='female')]['Age'].mean()
age_female_mrs = df[(df['Sex']=='female') & df['Name'].str.contains('Mrs')]['Age'].mean()
age_female_miss = df[(df['Sex']=='female') & df['Name'].str.contains('Miss')]['Age'].mean()

df.loc[(df['Sex'] == 'male') & df['Age'].isnull(), 'Age'] = age_male
df.loc[(df['Sex'] =='female') & df['Name'].str.contains('Mrs') & df['Age'].isnull(),'Age'] = age_female_mrs
df.loc[(df['Sex'] =='female') & df['Name'].str.contains('Miss') & df['Age'].isnull(),'Age'] = age_female_miss
df.loc[(df['Sex'] =='female') & df['Age'].isnull(),'Age'] = age_female

df_test.loc[(df_test['Sex']=='male') & df_test['Age'].isnull(),'Age'] = age_male
df_test.loc[(df_test['Sex']=='female') & df_test['Name'].str.contains('Mrs') & df_test['Age'].isnull(),'Age'] = age_female_mrs
df_test.loc[(df_test['Sex']=='female') & df_test['Name'].str.contains('Miss') & df_test['Age'].isnull(),'Age'] = age_female_miss
df_test.loc[(df_test['Sex']=='female') & df_test['Age'].isnull(),'Age'] = age_female


# for Pclass
fareP1 = df[df['Pclass'] == 1]['Fare'].median()
fareP2 = df[df['Pclass'] == 2]['Fare'].median()
fareP3 = df[df['Pclass'] == 3]['Fare'].median()

df_test.loc[(df_test['Pclass'] == 1) & df_test['Fare'].isnull(), 'Fare'] = fareP1
df_test.loc[(df_test['Pclass'] == 2) & df_test['Fare'].isnull(), 'Fare'] = fareP2
df_test.loc[(df_test['Pclass'] == 3) & df_test['Fare'].isnull(), 'Fare'] = fareP3


# AgeとFareをビン分割
df['Age'] = pd.qcut(df['Age'], 10, labels = False)
df_test['Age'] = pd.qcut(df_test['Age'], 10, labels = False)

df['Fare'] = pd.qcut(df['Fare'], 13, labels = False)
df_test['Fare'] = pd.qcut(df_test['Fare'], 13, labels = False)


# Cabinの値が空白か否かの列を追加。Nullなら1、そうでなければ0
df['Cabin_Null'] = df['Cabin'].isnull()*1
df_test['Cabin_Null'] = df_test['Cabin'].isnull()*1

# Cabinは削除
df.drop(['Cabin'], axis=1, inplace=True)
df_test.drop(['Cabin'], axis=1, inplace=True)


# Embarkedの欠損はSで補完する
df["Embarked"].value_counts()

df['Embarked'].fillna('S', inplace=True)
df_test['Embarked'].fillna('S', inplace=True)

# Embarkedをラベルエンコーディング
df['Embarked'].replace({"S": 0, "Q": 1, "C": 2}, inplace=True)
df_test['Embarked'].replace({"S": 0, "Q": 1, "C": 2}, inplace=True)


# Nameに含まれる敬称の情報から`Title`列を作成。ラベルエンコーディングする。
df['Title'] = df['Name'].apply(lambda x: 0 if 'Mr.' in x else (1 if 'Master.' in x else (2 if 'Miss.' in x else (3 if 'Mrs.' in x else 4))))
df_test['Title'] = df_test['Name'].apply(lambda x: 0 if 'Mr.' in x else (1 if 'Master.' in x else (2 if 'Miss.' in x else (3 if 'Mrs.' in x else 4))))

# Nameは削除
df.drop('Name', axis=1, inplace=True)
df_test.drop('Name', axis=1, inplace=True)


# 同一Ticketナンバーの人が何人いるかを特徴量として抽出
df["Ticket"].value_counts()

condition = df["Ticket"].value_counts() > 1
count = condition.value_counts()
print(count)

ticket_count = dict(df['Ticket'].value_counts())
ticket_count_test = dict(df_test['Ticket'].value_counts())

df['TicketGroup'] = df['Ticket'].map(ticket_count)
df_test['TicketGroup'] = df_test['Ticket'].map(ticket_count_test)

# カウントが2～4の人は生存率が高いため0, それ以外を1とする
df["TicketGroup"].value_counts()
df.groupby("TicketGroup")["Survived"].mean()

df.loc[(df['TicketGroup']>=2) & (df['TicketGroup']<=4), 'Ticket_Label'] = 0
df['Ticket_Label'].fillna(1, inplace=True)
df_test.loc[(df_test['TicketGroup']>=2) & (df_test['TicketGroup']<=4), 'Ticket_Label'] = 0
df_test['Ticket_Label'].fillna(1, inplace=True)

# Ticketと不要な列を削除
df.drop(['Ticket','TicketGroup'], axis=1, inplace=True)
df_test.drop(['Ticket','TicketGroup'], axis=1, inplace=True)


# Sexの数値化
df.replace({'Sex': {'male': 0, 'female': 1}}, inplace=True)
df_test.replace({'Sex': {'male': 0, 'female': 1}}, inplace=True)


# Family = SibSp + Parch を追加。人数が0なら0、2～4なら1、それ以上は2にする。人数による生存率の違いのため。
df['Family'] = (df['SibSp'] + df['Parch']).apply(lambda x: 0 if x == 0 else (1 if 2<=x<=4 else 2))
df_test['Family'] = (df_test['SibSp'] + df_test['Parch']).apply(lambda x: 0 if x == 0 else (1 if 2<=x<=4 else 2))

# reference
# df["FamilySize"] = df['SibSp'] + df['Parch']
# df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch']
# df.groupby("FamilySize")["Survived"].mean()

# SibSp Parchは削除
df.drop(['SibSp','Parch', 'FamilySize'], axis=1, inplace=True)
df_test.drop(['SibSp','Parch', 'FamilySize'], axis=1, inplace=True)


# Baselineの作成
X = df.iloc[:, 2:].values
y = df.iloc[:, 1].values

X_test = df_test.iloc[:, 1:].values

# 疑似訓練データと疑似テストデータの作成
MyRand = 0
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=MyRand)

from skopt import BayesSearchCV

# 決定木(ベースラインモデル)の調整
rfc = RandomForestClassifier(n_estimators = 150, random_state = MyRand)

# 標準化
sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_valid_std = sc.transform(X_valid)

np.int = int

# ハイパーパラメータの探索範囲を指定
param_space = {'max_depth': (2,5),
              # 'max_leaf_nodes': (5,10),
              'min_samples_leaf': (1,5),
              'min_samples_split': (2,5)}

# ベイズ最適化の実行
rfc = BayesSearchCV(rfc, param_space, n_iter=50, cv=5, n_jobs=-1, random_state=MyRand)
rfc.fit(X_train_std, y_train)

# 最適なハイパーパラメータを表示
print('Best Parameters: {}'.format(rfc.best_params_))
print('CV Score: {}\n'.format(round(rfc.best_score_, 3)))

rfc = rfc.best_estimator_

print('Random Forest Classifier')
print('Train Score: {}'.format(round(rfc.score(X_train_std, y_train), 3)))
print(' Test Score: {}'.format(round(rfc.score(X_valid_std, y_valid), 3)))

# ランダムフォレストで重視されている特徴量の可視化
# https://happy-analysis.com/python/python-topic-tree-importance.html

importances = rfc.feature_importances_
feature_names = df.iloc[:, 2:].columns

feature_importance_df = pd.DataFrame({'Feature':feature_names, 'Importance':importances})

feature_importance_df = feature_importance_df.sort_values(by='Importance',ascending=False)

plt.figure(figsize=(10,6), facecolor='gray')
plt.rcParams["font.size"] = 18
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
# plt.savefig("Feature Importance.png", bbox_inches='tight')
plt.show()

# 多層パーセプトロン
# https://zenn.dev/nekoallergy/articles/sklearn-nn-mlpclf02

# 標準化
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_valid_std = sc.transform(X_valid)

mlpc = MLPClassifier(hidden_layer_sizes=(11,47), random_state=MyRand)
mlpc.fit(X_train_std, y_train)

print('Multilayer Perceptron Classifier')
print('Train Score: {}'.format(round(mlpc.score(X_train_std, y_train), 3)))
print('Test Score: {}'.format(round(mlpc.score(X_valid_std, y_valid), 3)))

# ロジスティック回帰
# スケーリング
# https://qiita.com/ttskng/items/2a33c1ca925e4501e609
# https://hawk-tech-blog.com/python-machine-learning-basic-scaling/

# 標準化を行う
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_valid_std = sc.transform(X_valid)

# ロジスティック回帰のハイパーパラメータ探索空間
param_space = {'C': (1e-5, 1e+3, 'log-uniform')}

# ベイズ最適化の実行
lr = LogisticRegression(random_state=MyRand)
lr = BayesSearchCV(lr, param_space, cv=5, n_iter=50, n_jobs=-1, random_state=MyRand)
lr.fit(X_train_std, y_train)

# 最適なハイパーパラメータを表示
print('Best Parameters: {}'.format(lr.best_params_))
print('CV Score: {}\n'.format(round(lr.best_score_, 3)))

lr = lr.best_estimator_
lr.fit(X_train_std, y_train)

print('Logistic Regression')
print('Train Score: {}'.format(round(lr.score(X_train_std, y_train), 3)))
print(' Test Score: {}'.format(round(lr.score(X_valid_std, y_valid), 3)))

# サポートベクターマシンのパラメータ調整
# https://chat.openai.com/c/38b179d8-a042-4a9f-9bde-f8e54a171a5d
svc = SVC(probability=True, random_state=MyRand)

# 標準化
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_valid_std = sc.transform(X_valid)

# ハイパーパラメータの探索範囲を指定
param_space = {'C': (1e-3, 1e+3, 'log-uniform'),
               'gamma': (1e-3, 1e+1, 'log-uniform')}

# ベイズ最適化の実行
svc = BayesSearchCV(svc, param_space, n_iter=50, cv=5, n_jobs=-1, random_state=MyRand)
svc.fit(X_train_std, y_train)

# 最適なハイパーパラメータを表示
print('Best Parameters: {}'.format(svc.best_params_))
print('CV Score: {}\n'.format(round(svc.best_score_, 3)))

svc = svc.best_estimator_
svc.fit(X_train_std,y_train)

print('Support Vector Machine')
print('Train Score: {}'.format(round(svc.score(X_train_std, y_train), 3)))
print(' Test Score: {}'.format(round(svc.score(X_valid_std, y_valid), 3)))

# k-NN
# 標準化
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_valid_std = sc.transform(X_valid)

# グラフ描画用のリストを用意
training_accuracy = []
test_accuracy = []

for n_neighbors in range(1, 30):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train_std, y_train)
    training_accuracy.append(knn.score(X_train_std, y_train))
    test_accuracy.append(knn.score(X_valid_std, y_valid))

# グラフを描画
plt.plot(range(1, 30), training_accuracy, label='Training')
plt.plot(range(1, 30), test_accuracy, label='Test')
plt.ylabel('Accuracy')
plt.xlabel('n_neighbors')
plt.legend()

# 調整済みのKNN

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train_std, y_train)

print('k-Nearest Neighbor')
print('n = ',knn.n_neighbors)
print('Train Score: {}'.format(round(knn.score(X_train_std, y_train), 3)))
print(' Test Score: {}'.format(round(knn.score(X_valid_std, y_valid), 3)))


# テストデータの予測
# データの正規化
sc = StandardScaler()
sc.fit(X)
X_std = sc.transform(X)
X_test_std = sc.transform(X_test)

# モデルの再学習
rfc.fit(X_std, y)
mlpc.fit(X_std, y)
lr.fit(X_std, y)
svc.fit(X_std, y)
knn.fit(X_std, y)

# テストデータの回答を予測
rfc_pred = rfc.predict_proba(X_test_std)
mlpc_pred = mlpc.predict_proba(X_test_std)
lr_pred = lr.predict_proba(X_test_std)
svc_pred = svc.predict_proba(X_test_std)
knn_pred = knn.predict_proba(X_test_std)

# 平均を最終的な回答とする
pred_proba = (rfc_pred + mlpc_pred + lr_pred + svc_pred + knn_pred) / 5
pred =pred_proba.argmax(axis=1)

# 提出用データ作成

path = r""

submission = pd.read_csv(path + r"\gender_submission.csv")
submission['Survived'] = pred

submission.to_csv('submission.csv',index=False)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

df_wine = pd.read_csv('https://archive.ics.uci.edu/'
'ml/machine-learning-databases/'
'wine/wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']

print('Class labels', np.unique(df_wine['Class label']))
print(df_wine.head())
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test =train_test_split(X, y,test_size=0.1,random_state=0,stratify=y)

kfold = StratifiedKFold(n_splits=10,random_state=1).split(X_train,y_train)


for i in [20, 50, 100, 200, 1000]:
    forest = RandomForestClassifier(n_estimators=i,random_state=1)
    
    
    scores = cross_val_score(estimator=forest,X=X,y=y,cv=10)
    print('Score=',np.mean(scores),'in n_estimators=', i )
    
    
feat_labels = df_wine.columns[1:]
forest2=forest = RandomForestClassifier(n_estimators=200,random_state=1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,feat_labels[indices[f]],importances[indices[f]]))
    
plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]),importances[indices],align='center')
plt.xticks(range(X_train.shape[1]),feat_labels, rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

print("My name is {Jiayong Huang}")
print("My NetID is: {jiayong2}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

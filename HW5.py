#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 16:13:41 2018

@author: jiayonghuang
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.decomposition import KernelPCA
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from mlxtend.plotting import plot_decision_regions
from sklearn.svm import SVC

_
pd.set_option("display.max_columns", 100)
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                      header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

print(df_wine.head())

# Splitting the data into 80% training and 20% test subsets.
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                     stratify=y,
                     random_state=42)

# Standardizing the data.

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

df_wine.info()
print(df_wine.describe())

cm = np.corrcoef(df_wine.values.T)
#sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=False,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 6},
                 yticklabels=df_wine.columns,
                 xticklabels=df_wine.columns)

plt.tight_layout()
# plt.savefig('images/10_04.png', dpi=300)
plt.show()


#beseline
#Logistic	
lr = LogisticRegression()
lr=lr.fit(X_train_std, y_train)
y_pred=lr.predict(X_test_std)
y_pred_train=lr.predict(X_train_std)

print('Accuracy of baseline LogisticRegression test: %.2f' % accuracy_score(y_test, y_pred))
print('Accuracy of baseline LogisticRegression train: %.2f' % accuracy_score(y_train, y_pred_train))

#svm
svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train_std, y_train)
y_pred=lr.predict(X_test_std)
y_pred_train=lr.predict(X_train_std)

print('Accuracy of baseline SVM test: %.2f' % accuracy_score(y_test, y_pred))
print('Accuracy of baseline SVM train: %.2f' % accuracy_score(y_train, y_pred_train))



#PCA transform
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
#Logistic
lr=lr.fit(X_train_pca, y_train)
y_pred=lr.predict(X_test_pca)
y_pred_train=lr.predict(X_train_pca)

print('Accuracy of PCA transform test: %.2f' % accuracy_score(y_test, y_pred))
print('Accuracy of PCA transform  train: %.2f' % accuracy_score(y_train, y_pred_train))

#svm
svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train_pca, y_train)
y_pred=lr.predict(X_test_pca)
y_pred_train=lr.predict(X_train_pca)
print('Accuracy of PCA transform SVM test: %.2f' % accuracy_score(y_test, y_pred))
print('Accuracy of PCA transform SVM train: %.2f' % accuracy_score(y_train, y_pred_train))

#LDA transform
	
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)
#Logistic
lr=lr.fit(X_train_lda, y_train)
y_pred=lr.predict(X_test_lda)
y_pred_train=lr.predict(X_train_lda)

print('Accuracy of LDA transform test: %.2f' % accuracy_score(y_test, y_pred))
print('Accuracy of LDA transform train: %.2f' % accuracy_score(y_train, y_pred_train))
#svm
svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train_lda, y_train)
y_pred=lr.predict(X_test_lda)
y_pred_train=lr.predict(X_train_lda)
print('Accuracy of LDA transform SVM test: %.2f' % accuracy_score(y_test, y_pred))
print('Accuracy of LDA transform SVM train: %.2f' % accuracy_score(y_train, y_pred_train))

#kPCA transform
scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_train_kpca = scikit_kpca.fit_transform(X_train_std)
X_test_kpca=scikit_kpca.transform(X_test_std)
#Logistic
lr=lr.fit(X_train_kpca, y_train)
y_pred=lr.predict(X_test_kpca)
y_pred_train=lr.predict(X_train_kpca)
print('Accuracy of kPCA transform test: %.2f' % accuracy_score(y_test, y_pred))
print('Accuracy of kPCA transform train: %.2f' % accuracy_score(y_train, y_pred_train))

#svm
svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train_kpca, y_train)
y_pred=lr.predict(X_test_kpca)
y_pred_train=lr.predict(X_train_kpca)
print('Accuracy of kPCA transform SVM test: %.2f' % accuracy_score(y_test, y_pred))
print('Accuracy of kPCA transform SVM train: %.2f' % accuracy_score(y_train, y_pred_train))


print("My name is Jiayong Huang")
print("My NetID is: Jiayong2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.utils import resample
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

# Question 1
train_scores = []
test_scores = []

for seed in range(1, 11, 1):
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed, stratify=y)
    
    tree = DecisionTreeClassifier(criterion='gini', 
                              max_depth=4, 
                              random_state=1)
    tree.fit(X_train, y_train)
    train_scores.append(tree.score(X_train, y_train))
    test_scores.append(tree.score(X_test, y_test))

train_mean = np.mean(train_scores)
train_std = np.std(train_scores)
test_mean = np.mean(test_scores)
test_std = np.std(test_scores)

from sklearn.model_selection import cross_val_score
tree = DecisionTreeClassifier(criterion='gini', 
                              max_depth=4, 
                              random_state=1)

cv_scores = cross_val_score(tree, X, y, cv=10)

cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)




print("My name is Jiayong Huang")
print("My NetID is: Jiayong2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
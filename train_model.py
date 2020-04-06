# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 11:20:28 2020

@author: Lucky
"""
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from preprocess import preprocess as pp
import pandas as pd

df = pd.read_csv(input())
x, y = pp(df)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
#print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

from sklearn.linear_model import Perceptron
ppn = Perceptron()
ppn.fit(x_train, y_train)
ppn_pred = ppn.predict(x_test)
print(accuracy_score(ppn_pred, y_test))
print(confusion_matrix(ppn_pred, y_test))

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
knn_pred = knn.predict(x_test)
print(accuracy_score(knn_pred, y_test))
print(confusion_matrix(knn_pred, y_test))

from sklearn.linear_model import LogisticRegression
lor = LogisticRegression()
lor.fit(x_train, y_train)
lor_pred = lor.predict(x_test)
print(accuracy_score(lor_pred, y_test))
print(confusion_matrix(lor_pred, y_test))

from sklearn.svm import SVC
svm = SVC()
svm.fit(x_train, y_train)
svm_pred = svm.predict(x_test)
print(accuracy_score(svm_pred, y_test))
print(confusion_matrix(svm_pred, y_test))

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train, y_train)
nb_pred = nb.predict(x_test)
print(accuracy_score(nb_pred, y_test))
print(confusion_matrix(nb_pred, y_test))

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
print(rf.fit(x_train, y_train))
y_predict = rf.predict(x_test)
print(accuracy_score(y_predict, y_test))
print(confusion_matrix(y_predict, y_test))

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = "gini")
print(dt.fit(x_train, y_train))
y_predict = dt.predict(x_test)
print(accuracy_score(y_predict, y_test))
print(confusion_matrix(y_predict, y_test))


dt = DecisionTreeClassifier(criterion = "entropy")
print(dt.fit(x_train, y_train))
y_predict = dt.predict(x_test)
print(accuracy_score(y_predict, y_test))
print(confusion_matrix(y_predict, y_test))

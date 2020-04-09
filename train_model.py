# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 11:20:28 2020

@author: Lucky
"""
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from preprocess import *
import pandas as pd
import matplotlib.pyplot as plt

# taking file name as input and reading file as DataFrame 
df = pd.read_csv(input("Enter train file name: "))

# preprocessing the dataset
x, y , c = pp(df)   # returning features, target and c ('Course')

# ploting predectied data
def plot_scatter(t, y, name):
    plt.scatter(t, y, color = 'r')
    plt.title(name)
    plt.xlabel('Target')
    plt.ylabel('Predicted')
    plt.show()
    
# spliting the data for training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

# Applying KNN algo
print('KNN')
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(algorithm = 'auto', p = 2)
knn.fit(x_train, y_train)
def KNN(x_test, y_test):
    knn_pred = knn.predict(x_test)
    print(accuracy_score(knn_pred, y_test))
    print(confusion_matrix(knn_pred, y_test))
    plot_scatter(y_test, knn_pred, 'KNeighborsClassifier')
    return knn_pred

# applying SVC algo
print('SVC')
from sklearn.svm import SVC
svm = SVC(kernel = 'linear', gamma = 'auto' ,  C = 1.0)
svm.fit(x_train, y_train)
def SVC_(x_test, y_test):
    svm_pred = svm.predict(x_test)
    print(accuracy_score(svm_pred, y_test))
    print(confusion_matrix(svm_pred, y_test))
    plot_scatter(y_test, svm_pred, 'SVC')
    return svm_pred

# Applying Random forest
print('RFC')
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 100, criterion = 'gini')
print(rf.fit(x_train, y_train))
def RFC(x_test, y_test):
    rf_pred = rf.predict(x_test)
    print(accuracy_score(rf_pred, y_test))
    print(confusion_matrix(rf_pred, y_test))
    plot_scatter(y_test, rf_pred, 'RandomForestClassifier')
    return rf_pred

# analyzing the diffuculty of course
def analyze(c , y):
    d = pd.DataFrame(c)
    d['Grade'] = y
    l = d['Course'].unique()
    m, n = dict(), dict()
    for i in l:
        x = d.loc[d['Course'] == i]
        p = x['Grade'].unique()
        #p.sort()
        k = list(x.loc[:, ['Grade']].values)
        n[i] = list(x.iloc[:, 1].values)
        s = []
        for j in p:
            s.append(k.count(j))
        print(i, p, s)
        m[i] = int(p[np.argmax(s)])
        #m[i] = int(np.mean(s))
    return m, n

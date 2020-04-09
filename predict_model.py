# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 11:21:05 2020

@author: Lucky
"""

from preprocess import preprocess as pp
from train_model import RFC, KNN, SVC_
from train_model import analyze as ana
import pandas as pd
import matplotlib.pyplot as plt

# taking file name as input and reading file as DataFrame 
df = pd.read_csv(input('Enter Predict file name:'))

# preprocessing the dataset
x, y , c = pp(df)   # returning features, target and c ('Course')
print(x.shape, y.shape)   # printing the shape of preprocessed data 'x' and 'y'

# Ploting analyzed data
def plot_(x, name):
    plt.bar(list(x.keys()), x.values(), color='g')
    plt.title(name)
    plt.xlabel('Course')
    plt.ylabel('Diff')
    plt.show()

# taking input for particular Algo
inp = int(input('Enter 0 for RFC/ Enter 1 for SVC/ Enter 2 for KNN/ Enter 3 for all: '))
if inp == 0:
    # Random Forest
    pred = RFC(x, y)
    m, n = ana(c, pred)
    plot_(m, 'RFC')
elif inp == 1:
    # svm
    pred = SVC_(x ,y)
    m, n = ana(c, pred)
    plot_(m, 'SVC')
elif inp == 2:
    # knn
    pred = KNN(x, y)
    m, n = ana(c, pred)
    plot_(m, 'KNN')
elif inp == 3:
    # all algo
    pred1 = RFC(x, y)
    m, n = ana(c, pred1)
    plot_(m, 'RFC')
    pred2 = SVC_(x, y)
    m, n = ana(c, pred2)
    plot_(m, 'SVC')
    pred3 = KNN(x, y)
    m, n = ana(c, pred3)
    plot_(m, 'KNN')
else:
    print('Invalid Input')
# DATA-FINAL1.csv

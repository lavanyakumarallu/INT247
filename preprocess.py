# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 09:00:20 2020

@author: Lucky
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
def preprocess(df):
    df = df.loc[df['MHRDName'] == 'Diploma in Electronics and Communication Engineering']
    data = df.loc[:, ['Grade', 'CA_100', 'MTT_50', 'ETT_100','ETP_100', 'Course_Att']]
    c = df.loc[:, ['Course']]
    map1={'O':10,'A+':9,'A':8,'B+':7,'B':6,'C':5,'D':4,'E':3,'F':2}
    data['Grade']=data['Grade'].map(map1)
    x = data.iloc[:, 1:]
    y = data.loc[:, ['Grade']]
    si = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
    x = si.fit_transform(x)
    x = StandardScaler().fit_transform(x)
    df['Grade'].value_counts().plot(kind='barh', title = 'Grade')
    plt.show()
    df['MTT_50'].value_counts().plot(kind='hist', title = 'MTT_50')
    plt.show()
    df['ETT_100'].value_counts().plot(kind='hist', title = 'ETT_100')
    plt.show()
    df['ETP_100'].value_counts().plot(kind='hist', title = 'ETP_100')
    plt.show()
    df['CA_100'].value_counts().plot(kind='hist', title = 'CA_100')
    plt.show()
    return x, y, c

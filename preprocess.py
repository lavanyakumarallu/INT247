# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 09:00:20 2020

@author: Lucky
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
def preprocess(df):
    df = df.loc[df['MHRDName'] == 'Diploma in Electronics and Communication Engineering']
    data = df.loc[:, ['Grade', 'CA_100', 'MTT_50', 'ETT_100','ETP_100', 'Course_Att']]
    x = data.iloc[:, 1:]
    y = LabelEncoder().fit_transform(data['Grade'])
    si = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
    x = si.fit_transform(x)
    x = StandardScaler().fit_transform(x)
    return x, y
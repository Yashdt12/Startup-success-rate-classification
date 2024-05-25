#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 19:56:37 2023

@author: yash
"""

import pandas as pd

df = pd.read_csv('startup.csv')
X = df.drop(['Unnamed: 0',
             'state_code',
             'latitude',
             'longitude',
             'zip_code',
             'id',
             'city',
             'Unnamed: 6',
             'name',
             'labels',
             'founded_at',
             'closed_at',
             'first_funding_at',
             'last_funding_at',
             'state_code.1',
             'category_code',
             'object_id',
             'status'],axis = 1)
mean1 = X['age_first_milestone_year'].mean()
mean2 = X['age_last_milestone_year'].mean()
X['age_first_milestone_year'].fillna(mean1, inplace = True)
X['age_last_milestone_year'].fillna(mean2, inplace = True)
y = df['status']

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.decomposition import PCA
principal = PCA(n_components = 2)
dfx_pca = principal.fit(X)
dfx_trans = principal.transform(X)
dfx_trans = pd.DataFrame(data=dfx_trans)

import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.scatter(dfx_trans[0],dfx_trans[1],c=y,edgecolors='k',alpha=0.75,s=150)
plt.grid(True)
plt.title("Class separation using first two principal components",fontsize=20)
plt.xlabel("Principal component-1",fontsize=15)
plt.ylabel("Principal component-2",fontsize=15)
plt.show()
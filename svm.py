#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 17:22:51 2023

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
y= df['status']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size = 0.2, random_state = 33)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

from sklearn.svm import SVC
svc = SVC(C = 1.0, random_state = 1, kernel = 'linear')
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test,y_pred)
print("Confustion matrix:\n",cm)
print("Accuracy score: ",ac)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 12:36:06 2023

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

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size = 0.2, random_state = 20)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
y_pred = dtree.predict(X_test)
print("Classification report - \n", classification_report(y_test,y_pred))
cm = confusion_matrix(y_test, y_pred)
print("Confustion matrix: \n",cm)
ac = accuracy_score(y_test, y_pred)
print("Accuracy score: ",ac)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
plt.figure(figsize=(5,5))
sns.heatmap(data=cm,linewidths=.5, annot=True,square = True, cmap = 'Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(dtree.score(X_test, y_test))
plt.title(all_sample_title, size = 15)
plt.figure(figsize = (10,10))
dec_tree = plot_tree(decision_tree=dtree, feature_names = X.columns ,class_names =['0','1'] ,
                     filled = True ,precision = 4, rounded = True)
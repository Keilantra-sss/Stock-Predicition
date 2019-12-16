#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 22:54:32 2019

@author: keilantra
"""

import os
import numpy as np 
import pandas as pd 
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import recall_score
from sklearn import metrics

# Import data
ticker='PCG'
input_dir = r'/Users/keilantra/Desktop'
ticker_file = os.path.join(input_dir, ticker + '.csv')
plot_dir = r'/Users/keilantra/Desktop'
try:   
    df = pd.read_csv(ticker_file)
except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)  
    
# Data preparation
start_date='2017-01-01'
df = df[df['Date'] >= start_date]
df = df.reset_index()

# The 15th week in 2017 and 12nd week in 2018 have no Friday
# Insert the Thursday data to the dataframe
df2 = df[df['Weekday'] == 'Friday']
df2.loc[1] = df.iloc[70,]
df2.loc[2] = df.iloc[311,]
df2 = df2.sort_values(by = ['Year','Month','Day'])
df2_17 = df2[df2['Year'] == 2017].reset_index()
df2_18 = df2[df2['Year'] == 2018].reset_index()

# Calculate the μ and σ for year 2017 
df2017 = df[df['Year'] == 2017]
df2017 = df2017.groupby(['Week_Number','Label'])['Adj Close'].agg([np.mean,np.std])
df2017 = df2017.reset_index()
# Insert the Adj Close to year 2017
df2017['Adj Close'] = df2_17['Adj Close'] 
del df2_17

# Calculate the μ and σ for year 2018
df2018 = df[df['Year'] == 2018]
df2018 = df2018.groupby(['Week_Number','Label'])['Adj Close'].agg([np.mean,np.std])
df2018 = df2018.reset_index()
# Insert the Adj Close to year 2018
df2018['Adj Close'] = df2_18['Adj Close'] 
del df2_18


# Preprocessing data
feature = ['mean','std']
X = df2017[feature].values
le = LabelEncoder()
y = le.fit_transform(df2017['Label'].values)
X2 = df2018[feature].values
y2 = le.fit_transform(df2018['Label'].values)
# For y: 0 refers to green, 1 refers to red

# Perform decision tree classifier
tree_classifier = tree.DecisionTreeClassifier(criterion = 'entropy')
tree_classifier = tree_classifier.fit(X,y)
y_pred = tree_classifier.predict(X2)


# Question 1: Accuracy for 2018
accuracy = metrics.accuracy_score(y2, y_pred)
print('The accuracy for for year 2018 is',round(accuracy,2))


# Question 2: Confusion matrix for year 2018
cm = confusion_matrix(y2, y_pred)
print('The confusion matrix is:\n',cm)


# Question 3 & Question 4: TPR & TNR
print(classification_report(y2, y_pred))
TPR = recall_score(y2, y_pred)
# TPR = cm[1,1]/(cm[1,1] + cm[1,0])
print('The true positive rate (sensitivity or recall) for 2018 is',round(TPR,2))
TNR = cm[0,0]/(cm[0,0] + cm[0,1]) 
print('The true negative rate (speciﬁcity) for 2018 is',round(TNR,2))


# Question 5: Comparison
# Buy-and-hold strategy
share = 100/df2018['Adj Close'][0]
value = share * df2018['Adj Close'][51]
print('For Buy-and-hold strategy, the final amount is $',round(value,2))

# New strategy
# Based on decision tree
share = 100/df2018['Adj Close'][0]
cash = 0
position = 0
for i in range(len(y_pred)-1):
    if y_pred[i+1] == 1:
        if position != 0:
            cash += share * df2018['Adj Close'][i]
            share = 0
            position = 0
    elif y_pred[i+1] == 0:
         if position == 0:
             share += cash/df2018['Adj Close'][i]
             cash = 0
             position = 1
value_new = cash + share * df2018['Adj Close'][51]
print('For the trading strategy using decision tree, the final amount is $',round(value_new,2))

# Comparison
if value_new > value:
    print('The decision tree results in a larger amount at the end of the year.')
elif value_new < value:
    print('The buy-and-hold results in a larger amount at the end of the year.')
else:
    print('The two strategies result in a larger amount at the end of the year.')


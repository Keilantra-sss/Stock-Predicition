#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 13:10:36 2019

@author: keilantra
"""

import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, recall_score

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
#y = df2017['Label'].values
y = le.fit_transform(df2017['Label'].values)

X2 = df2018[feature].values
#y2 = df2018['Label'].values
y2 = le.fit_transform(df2018['Label'].values)


# Find the best combination of N and d 
result = pd.DataFrame(index=range(1,11), columns=range(1,6))
for N in range(1,11):
    for d in range(1,6):
        clf = RandomForestClassifier(n_estimators=N, max_depth=d, criterion='entropy')
        clf.fit(X,y)
        y_pred = clf.predict(X2)  
        error = np.mean(y_pred != y2)
        result.iloc[N-1, d-1] = error
        
# Question 1
# Plot the error rate
best = sorted(result.min())[0]
for N in range(1,11):
    for d in range(1,6):
        if result.iloc[N-1,d-1] > best:
            col = 'green'
        else:
            col = 'red'
        plt.scatter(N, d ,s=result.iloc[N-1,d-1]*100, color=col)
plt.show()

# Find the best combination
for indexs in result.index:
    for i in range(len(result.loc[indexs].values)):
        if result.loc[indexs].values[i] == best:
            N = indexs
            d = i+1
print('The best combination is N=',N,'d=',d)
            

# Perform the random forest classifier using optimal values
clf = RandomForestClassifier(n_estimators=N, max_depth=d, criterion='entropy')
clf.fit(X,y)
y_pred = clf.predict(X2) 


# Question 2: Confusion matrix for year 2018
cm = confusion_matrix(y2, y_pred)
print('The confusion matrix for random forest classifier is:\n',cm)


# Question 3 & 4: TPR & TNR
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
print('For the trading strategy using random forest, the final amount is $',round(value_new,2))

# Comparison
if value_new > value:
    print('The random forest results in a larger amount at the end of the year.')
elif value_new < value:
    print('The buy-and-hold results in a larger amount at the end of the year.')
else:
    print('The two strategies result in a larger amount at the end of the year.')

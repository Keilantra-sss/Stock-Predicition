#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 22:30:07 2019

@author: keilantra
"""

import os
import numpy as np 
import pandas as pd 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
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
X = StandardScaler().fit_transform(X)
le = LabelEncoder()
y = df2017['Label'].values
#y = le.fit_transform(df2017['Label'].values)

X2 = df2018[feature].values
X2 = StandardScaler().fit_transform(X2)
y2 = df2018['Label'].values
#y2 = le.fit_transform(df201['Label'].values)

# Perform LDA classifier
lda_classifier = LDA(n_components=2) 
lda_classifier.fit(X,y,)
y_pred1 = lda_classifier.predict(X2)

# Perform QDA classifier
qda_classifier = QDA() 
qda_classifier.fit(X,y)
y_pred2 = qda_classifier.predict(X2)


# Question 1: Equation for LDA
w1 = round(lda_classifier.coef_[0][0],4)
w2 = round(lda_classifier.coef_[0][1],4)
w0 = round(lda_classifier.intercept_[0],4)
print('The equation is: g(x) =',w0,'+',w1,'* mean +',w2,'* std')


# Question 2: Accuracy for year 2018
accuracy_lda = metrics.accuracy_score(y2, y_pred1)
print('The accuracy for LDA classifier for year 2018 is',round(accuracy_lda,2))
accuracy_qda = metrics.accuracy_score(y2, y_pred2)
print('The accuracy for QDA classifier for year 2018 is',round(accuracy_qda,2))
if accuracy_lda > accuracy_qda:
    print('The linear discriminant classifier is better.')
else:
    print('The quadratic discriminant classiﬁer is better.')


# Question 3: Confusion matrix for year 2018
cm_lda = confusion_matrix(y2, y_pred1)
cm_qda = confusion_matrix(y2, y_pred2)
print('The confusion matrix for LDA classifier is:\n',cm_lda)
print('The confusion matrix for QDA classifier is:\n',cm_qda)


# Question 4: TPR & TNR
# LDA classifier
print('For LDA classifier:')
print(classification_report(y2, y_pred1))
TPR_lda = cm_lda[1,1]/(cm_lda[1,1] + cm_lda[1,0])
print('The true positive rate (sensitivity or recall) for 2018 is',round(TPR_lda,2))
TNR_lda = cm_lda[0,0]/(cm_lda[0,0] + cm_lda[0,1]) 
print('The true negative rate (speciﬁcity) for 2018 is',round(TNR_lda,2))

# QDA classifier
print('For QDA classifier:')
print(classification_report(y2, y_pred2))
TPR_qda = cm_qda[1,1]/(cm_qda[1,1] + cm_qda[1,0])
print('The true positive rate (sensitivity or recall) for 2018 is',round(TPR_qda,2))
TNR_qda = cm_qda[0,0]/(cm_qda[0,0] + cm_qda[0,1]) 
print('The true negative rate (speciﬁcity) for 2018 is',round(TNR_qda,2))


# Question 5
# Buy-and-hold strategy
share = 100/df2018['Adj Close'][0]
value = share * df2018['Adj Close'][51]
print('For Buy-and-hold strategy, the final amount is $',round(value,2))

# New strategy
# Based on LDA
share = 100/df2018['Adj Close'][0]
cash = 0
position = 0
for i in range(len(y_pred1)-1):
    if y_pred1[i+1] == 'Red':
        if position != 0:
            cash += share * df2018['Adj Close'][i]
            share = 0
            position = 0
    elif y_pred1[i+1] == 'Green':
         if position == 0:
             share += cash/df2018['Adj Close'][i]
             cash = 0
             position = 1
value_lda = cash + share * df2018['Adj Close'][51]
print('For the trading strategy using LDA, the final amount is $',round(value_lda,2))

# Based on QDA
share = 100/df2018['Adj Close'][0]
cash = 0
position = 0
for i in range(len(y_pred2)-1):
    if y_pred2[i+1] == 'Red':
        if position != 0:
            cash += share * df2018['Adj Close'][i]
            share = 0
            position = 0
    elif y_pred2[i+1] == 'Green':
         if position == 0:
             share += cash/df2018['Adj Close'][i]
             cash = 0
             position = 1
value_qda = cash + share * df2018['Adj Close'][51]
print('For the trading strategy using QDA, the final amount is $',round(value_qda,2))

# Comparison
compare = [value, value_lda, value_qda]
if value_lda == max(compare):
    print('The LDA results in a larger amount at the end of the year.')
elif value_qda == max(compare):
    print('The QDA results in a larger amount at the end of the year.')
else:
    print('The buy-and-hold strategy results in a larger amount at the end of the year.')





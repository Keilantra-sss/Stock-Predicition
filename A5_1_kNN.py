#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 17:16:09 2019

@author: keilantra
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
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
df2017['Label'][df2017['Label'] == 'Green'] = 1
df2017['Label'][df2017['Label'] == 'Red'] = 0
y = df2017.loc[:, 'Label'].values  
le = LabelEncoder()
X = StandardScaler().fit_transform(X)
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=3)
# For y: 1 refers to green, 0 refers to red


# Perform kNN model
k = [3,5,7,9,11]
accuracy = []
for n in k:
    knn = KNeighborsClassifier(n_neighbors=n, metric='euclidean')
    knn.fit(X_train, y_train)
    pred_k = knn.predict(X_test)
    accuracy.append(metrics.accuracy_score(y_test, pred_k))
    # error.append(np.mean(pred_k != y_test))


# Question 1: Plot the accuracy for each value of k
plt.plot(k, accuracy, linestyle='dashed', marker='o')
plt.title('Accuracy for K Value')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
    

# Question 2: Find the optimal k value
accuracy_df = pd.DataFrame(accuracy, k)
optimal = accuracy_df.idxmax()
print('The optimal value of k for 2017 is',optimal[0])


# Question 3
# Update the knn model
knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(X_train, y_train)

X2 = df2018[feature].values
le = LabelEncoder()
df2018['Label'][df2018['Label'] == 'Green'] = 1
df2018['Label'][df2018['Label'] == 'Red'] = 0
y2 = le.fit_transform(df2018['Label'].values)
X2 = StandardScaler().fit_transform(X2)
y_pred = knn.predict(X2)

accuracy = metrics.accuracy_score(y2, y_pred)
print('The accuracy is',round(accuracy,2))


# Question 4: Confusion Matrix
cm = confusion_matrix(y2, y_pred)
print('The confusion matrix is:\n',cm)

# Question 5 & Question 6: TPR & TNR
print(classification_report(y2, y_pred))
TPR = recall_score(y2, y_pred)
# TPR = cm[1,1]/(cm[1,1] + cm[1,0])
print('The true positive rate (sensitivity or recall) for 2018 is',round(TPR,2))
TNR = cm[0,0]/(cm[0,0] + cm[0,1]) 
print('The true negative rate (speciﬁcity) for 2018 is',round(TNR,2))

    
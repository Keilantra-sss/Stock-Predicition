#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 20:57:50 2019

@author: keilantra
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

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
attribute = ['mean','std']
X = df2017[attribute].values
df2017['Label'][df2017['Label'] == 'Green'] = 1
df2017['Label'][df2017['Label'] == 'Red'] = 0
y = df2017.loc[:, 'Label'].values  
le = LabelEncoder()
X = StandardScaler().fit_transform(X)
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=369)
# For y: 1 refers to green, 0 refers to red
X2 = df2018[attribute].values
df2018['Label'][df2018['Label'] == 'Green'] = 1
df2018['Label'][df2018['Label'] == 'Red'] = 0
le = LabelEncoder()
y2 = le.fit_transform(df2018['Label'].values)
X2 = StandardScaler().fit_transform(X2)


# Define the Custom_knn   
class Custom_knn:
    def __init__(self, number_neighbots_k, distance_parameter_p):  
        self.number_neighbots_k = number_neighbots_k
        self.distance_parameter_p = distance_parameter_p   
        self.X = None
        self.Labels = None
    
    def __str__(self):  
        return self.number_neighbots_k, self.distance_parameter_p                
    
    def fit(self, X, Labels):
        self.X = X
        self.Labels = Labels      
        return self
    
    def predict(self, new_x):
        assert self.X is not None
        assert self.Labels is not None
        predict = []
        for b in range(len(new_x)):
            dist = []
            for a in self.X:
                distance = np.linalg.norm(new_x[b]-a, ord = self.distance_parameter_p)
                dist.append(distance)
            sort = np.argsort(dist)
            topK = [self.Labels[i] for i in sort[:self.number_neighbots_k]]
            votes = Counter(topK)
            y_pred = votes.most_common(1)[0][0]
            predict.append(y_pred)
        return predict
        
    def draw_decision_boundary(self, new_x):
        dist = []
        for a in self.X:
            distance = np.linalg.norm(new_x-a, ord = self.distance_parameter_p)
            dist.append(distance)
        sort = np.argsort(dist)
            
        for i in sort[:self.number_neighbots_k]:
            print('id:',i, 'Label:',self.Labels[i])
         
# Hamming distance
def hamming(S1, S2):
    assert len(S1) == len(S2)
    dist = 0
    for i in range(len(S1)):
        if S1[i] != S2[i]:
            dist += 1
    return dist


# Question 1
# Buy-and-hold strategy
df2018['BnH'] = 1 # 1 refers to green
dist1 = hamming(df2018['BnH'],y2)
plt.scatter(x=df2018['Week_Number'], y=df2018['BnH'], color='b' ,s=10)
plt.title('Buy-and-hold Strategy for Year 2018')
plt.xlabel('Week_Number')
plt.ylabel('Value of the Week for the Strategy')
plt.yticks(np.arange(0,2,1))
plt.legend(['Hamming Distance = %s' % dist1])
plt.show()

# True label strategy
dist2 = hamming(y2, y2)
plt.scatter(x=df2018['Week_Number'], y=y2, color='g', s=10)
plt.title('True Label Strategy for Year 2018')
plt.xlabel('Week_Number')
plt.ylabel('Value of the Week for the Strategy')
plt.yticks(np.arange(0,2,1))
plt.legend(['Hamming Distance = %s' % dist2])
plt.show()

# kNN strategy
# The optimal k value in the last assignment is k=3
# Perform kNN model for different p 
knn1 = Custom_knn(3,1)
knn1.fit(X_train, y_train)
y_pred1 = knn1.predict(X2)

knn2 = Custom_knn(3,1.5)
knn2.fit(X_train, y_train)
y_pred2 = knn2.predict(X2)

knn3 = Custom_knn(3,2)
knn3.fit(X_train, y_train)
y_pred3 = knn3.predict(X2)

# kNN (p=1) strategy 
dist3_1 = hamming(y_pred1, y2)
plt.scatter(x=df2018['Week_Number'], y=y_pred1, color='r', s=10)
plt.title('kNN (p=1) Strategy for Year 2018')
plt.xlabel('Week_Number')
plt.ylabel('Value of the Week for the Strategy')
plt.yticks(np.arange(0,2,1))
plt.legend(['Hamming Distance = %s' % dist3_1])
plt.show()

# kNN (p=1.5) strategy 
dist3_2 = hamming(y_pred2, y2)
plt.scatter(x=df2018['Week_Number'], y=y_pred2, color='c', s=10)
plt.title('kNN (p=1.5) Strategy for Year 2018')
plt.xlabel('Week_Number')
plt.ylabel('Value of the Week for the Strategy')
plt.yticks(np.arange(0,2,1))
plt.legend(['Hamming Distance = %s' % dist3_2])
plt.show()

# kNN (p=2) strategy 
dist3_3 = hamming(y_pred3, y2)
plt.scatter(x=df2018['Week_Number'], y=y_pred3, color='m', s=10)
plt.title('kNN (p=2) Strategy for Year 2018')
plt.xlabel('Week_Number')
plt.ylabel('Value of the Week for the Strategy')
plt.yticks(np.arange(0,2,1))
plt.legend(['Hamming Distance = %s' % dist3_3])
plt.show()


# Question 2
# Build the hypothesis
print('Hypothesis: matrixM[i][j] = FP + FN')
print('The matrixM[i][j] is equal to the sum of false positive and false negative of the confusion matrix.')

# Matrix M
M = np.zeros(shape = (5,5))
element = pd.DataFrame({'Buy-and-Hold':1, 'True Label':y2, 
                       'kNN_1':y_pred1, 'KNN_2':y_pred2, 'kNN_3':y_pred3})
for i in range(5):
    for j in range(5):
        M[i][j] = hamming(element.iloc[:,i], element.iloc[:,j])

# Confusion matrix
def cm(i,j):
    return confusion_matrix(element.iloc[:,i], element.iloc[:,j])

# Test the hypothesis
M2 = np.zeros(shape = (5,5))
for i in range(5):
    for j in range(5):
        if i !=j:
            M2[i][j] = cm(i,j)[0,1] + cm(i,j)[1,0]
        
if (M == M2).all():
    print('The hypothesis is correct: matrixM[i][j] = FP + FN')
else:
    print('The hypothesis is false.')


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 18:57:50 2019

@author: keilantra
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import Counter
from sklearn.model_selection import train_test_split

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
le = LabelEncoder()
df2018['Label'][df2018['Label'] == 'Green'] = 1
df2018['Label'][df2018['Label'] == 'Red'] = 0
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

# Build the label index
label_index = pd.DataFrame({'Buy-and-Hold':1, 'True Label':y2,
                        'kNN_1':y_pred1, 'KNN_2':y_pred2, 'kNN_3':y_pred3})
    
# Assume the trading strategy is the same as Assignment 4 (Trading with Labels)         
# Recall the strategy:
# If next week n + 1 is predicted with a ”green” label:
# If you have no position then buy your stock at the price Pn
# If you have a position, do nothing 
# If week n + 1 is predicted with a ”red” label:
# If you have no position, do nothing 
# if you have a position then sell it at the end of the day (at price Pn) 
# no short positions are to be taken            
            
def profit(label,price):
    share = 0  
    position = 0
    profit = 100
    # Assume strating with $100
    balance = [100,]
    for i in range(len(label)-1):
        if label[i+1] == 1:
            # 1 refers to green
            if position == 0:
                share = profit/price[i]
                profit = 0
                position = 1
        elif label[i+1] == 0:
            # 0 refers to red
            if position != 0:
                profit = share * price[i]
                share = 0
                position = 0
        balance.append(profit+share*price[i])
    return balance

# Calculate the weekly value
price = df2018['Adj Close']
p1 = profit(label_index.iloc[:,0], price)
p2 = profit(label_index.iloc[:,1], price)
p3 = profit(label_index.iloc[:,2], price)
p4 = profit(label_index.iloc[:,3], price)
p5 = profit(label_index.iloc[:,4], price)

# Compute the mean and the std
def mean(price):
    return round(np.mean(price),2)

def std(price):
    return round(np.std(price),2)

# Plot the portfolio growth
x_axis = df2018['Week_Number']
plt.title('Portfolio Growth for Different Strategy')
plt.xlabel('Week Number')
plt.ylabel('Value of the Portfolio')
plt.plot(x_axis, p1, label='Buy-and-hold: mean=%s, std=%d'%(mean(p1),std(p1)))
plt.plot(x_axis, p2, label='True Label: mean=%s, std=%d'%(mean(p2),std(p2)))
plt.plot(x_axis, p3, label='kNN_1:mean=%s, std=%d'%(mean(p3),std(p3)))
plt.plot(x_axis, p4, label='kNN_2:mean=%s, std=%d'%(mean(p4),std(p4)))
plt.plot(x_axis, p5, label='kNN_3:mean=%s, std=%d'%(mean(p5),std(p5)))
plt.legend()
plt.show()



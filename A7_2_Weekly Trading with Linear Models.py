#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 19:50:52 2019

@author: keilantra
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
df1 = df[df['Weekday'] == 'Friday']
df1.loc[1] = df.iloc[70,]
df1.loc[2] = df.iloc[311,]
df1 = df1.sort_values(by = ['Year','Month','Day'])

# Get useful columns and add necessary columns
df1 = df1[['Year', 'Week_Number', 'Adj Close','Label']]
col_name = df1.columns.tolist()
col_name.insert(4,'Predict')
df1= df1.reindex(columns = col_name)

df2017 = df1[df1['Year'] == 2017]
df2017.index = range(len(df2017))
df2018 = df1[df1['Year'] == 2018]
df2018.index = range(len(df2018))
del df1


# Construct the polynomials model 
W = [5,6,7,8,9,10,11,12]
def poly_fun(d): 
    accuracy = []
    for j in W:   
        for i in range(len(df2017)-j):   
            x = np.array(range(1,1+j))
            y = df2017.loc[0:j-1,'Adj Close']
    
            degree = d
            weights = np.polyfit(x, y, degree) 
            model = np.poly1d(weights) 
            y_pred = model(np.array([[j+1]])) 
            
            if y_pred > df2017.loc[i+j-1,'Adj Close']:
                df2017['Predict'][i+j] = 'Green'
            elif y_pred < df2017.loc[i+j-1,'Adj Close']:
                df2017['Predict'][i+j] = 'Red'
            else:
                df2017['Predict'][i+j] = df2017['Label'][i+j-1]
                            
        accuracy0 = np.mean(df2017['Predict'][j:] == df2017['Label'][j:])
        accuracy.append(accuracy0)
    return accuracy

def plot(d):
    x_axis = W
    y_axis = poly_fun(d)  
    plt.title('The accuracy for year 2017')
    plt.legend(['d = %s' % d])
    plt.xlabel('W')
    plt.ylabel('Accuracy')
    plt.plot(x_axis, y_axis)
    return plt.show() 


# Question 1: Plot the accuracy for year 2017
poly_fun1 = poly_fun(1)
poly_fun2 = poly_fun(2)
poly_fun3 = poly_fun(3)

x_axis = W
plt.title('The accuracy for year 2017')
plt.xlabel('W')
plt.ylabel('Accuracy')
plt.plot(x_axis, poly_fun1, label='d=1')
plt.plot(x_axis, poly_fun2, label='d=2')
plt.plot(x_axis, poly_fun3, label='d=3')
plt.legend()
plt.show() 


# Question 2
# Find the best W for each d
accuracy_df_1 = pd.DataFrame(poly_fun1,W)
W1 = accuracy_df_1.idxmax()[0]
accuracy_df_2 = pd.DataFrame(poly_fun2,W)
W2 = accuracy_df_2.idxmax()[0]
accuracy_df_3 = pd.DataFrame(poly_fun3,W)
W3 = accuracy_df_3.idxmax()[0]

print('For d=1, the best W is',W1)
print('For d=2, the best W is',W2)
print('For d=3, the best W is',W3)

# Predict labels for year 2018
def pred_model(W, d):  
    for i in range(len(df2018)-W):   
        x = np.array(range(1,1+W))
        y = df2018.loc[0:W-1,'Adj Close']
    
        degree = d
        weights = np.polyfit(x, y, degree) 
        model = np.poly1d(weights) 
        y_pred = model(np.array([[W+1]])) 
            
        if y_pred > df2018.loc[i+W-1,'Adj Close']:
            df2018['Predict'][i+W] = 'Green'
        elif y_pred < df2018.loc[i+W-1,'Adj Close']:
            df2018['Predict'][i+W] = 'Red'
        else:
            df2018['Predict'][i+W] = df2018['Label'][i+W-1]
    return df2018['Predict'][W:]

# For d=1, the best W is 12
pred_model1 = pred_model(W1,1)
# For d=2, the best W is 12
pred_model2 = pred_model(W2,2)
# For d=3, the best W is 8
pred_model3 = pred_model(W3,3)

accuracy_1 = np.mean(pred_model1 == df2018['Label'][W1:])
accuracy_2 = np.mean(pred_model2 == df2018['Label'][W2:])
accuracy_3 = np.mean(pred_model3 == df2018['Label'][W3:])

print('For d=1, the accuracy is',round(accuracy_1,2))
print('For d=2, the accuracy is',round(accuracy_2,2))
print('For d=3, the accuracy is',round(accuracy_3,2))


# Question 3: Confusion Matrix
y_1 = df2018['Label'][W1:]
y_pred_1 = pred_model1
y_2 = df2018['Label'][W2:]
y_pred_2 = pred_model2
y_3 = df2018['Label'][W3:]
y_pred_3 = pred_model3

print('For d=1, the confusion matrix for is:\n',confusion_matrix(y_1, y_pred_1))
print('For d=2, the confusion matrix for is:\n',confusion_matrix(y_2, y_pred_2))
print('For d=3, the confusion matrix for is:\n',confusion_matrix(y_3, y_pred_3))


# Question 4: Implementing three trading strategies
# Buy-and-hold strategy
share = 100/df2018['Adj Close'][0]
value = share * df2018['Adj Close'][51]
print('For Buy-and-hold strategy, the final amount is $',round(value,2))

# New strategy
# Bases on the label predicted by linear models with different d
# Assume no trading when no predicted label
def new_strategy(W, d):
    y_pred = pred_model(W,d)   
    share = 100/df2018['Adj Close'][0]
    cash = 0
    position = 0
    for i in range(W, len(df2018)-1):
        if y_pred[i+1] == 'Red':
            if position != 0:
                cash += share * df2018['Adj Close'][i]
                share = 0
                position = 0 
        elif y_pred[i+1] == 'Green':
            if position == 0:
                share += cash/df2018['Adj Close'][i]
                cash = 0
                position = 1
    value_new = cash + share * df2018['Adj Close'][51]
    return value_new

value1 = new_strategy(W1,1)
value2 = new_strategy(W2,2)
value3 = new_strategy(W3,3)
print('For the new strategy:')
print('When d=1, W=12, the final amount is $',round(value1,2))
print('When d=2, W=12, the final amount is $',round(value2,2))
print('When d=3, W=8, the final amount is $',round(value3,2))

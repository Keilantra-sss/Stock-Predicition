#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 18:12:19 2019

@author: keilantra
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

# Split the data
df2017 = df[df['Year'] == 2017]
df2018 = df[df['Year'] == 2018]

# Calculate the μ and σ  
# Assume the feature set is calculted by Return          
df2017 = df2017.groupby(['Week_Number','Label'])['Return'].agg([np.mean,np.std]) 
df2017 = df2017.reset_index()
df2018 = df2018.groupby(['Week_Number','Label'])['Return'].agg([np.mean,np.std]) 
df2018 = df2018.reset_index()

# Plot the data
plt.figure(figsize=(13,13))
x = df2017['mean']
y = df2017['std']
pid = df2017['Week_Number']

for xi, yi, pidi in zip(x,y,pid):
    plt.annotate(str(pidi), xy=(xi,yi))
    
plt.scatter(x, y, c=df2017['Label'],s=15)
plt.xlabel('mean (μ)')
plt.ylabel('standard deviation (σ)')
plt.show()


# Remove the point of Week_Number 2
df2017_rm = df2017.drop([1])
plt.scatter(x=df2017_rm['mean'], y=df2017_rm['std'], c=df2017_rm['Label'],s=15)
plt.xlabel('mean (μ)')
plt.ylabel('standard deviation (σ)')
plt.plot([0,0],[-0.01,0.06])
print('The equation is: x = 0')


df2017_rm = df2017_rm.reset_index()
del df2017_rm['index']
k = 0
for i in range(len(df2017_rm)):
    if (0-df2017_rm['mean'][i])*(0.06-df2017_rm['std'][i])-(-0.01-df2017_rm['std'][i])*(-0-df2017_rm['mean'][i]) > 0:
        if df2017_rm['Label'][i] == 'Red':
            k +=1
    else:
        if df2017_rm['Label'][i] == 'Green':
            k +=1

if k == len(df2017_rm):
    print('The line successfully separates the points.')
    print('Red points locate on left side, green points locate on right side.')


# Question 2
# Take this line and use it to assign labels for year 2
# The two points are the points of x=0 used in year 1
A = [0, -0.01]
B = [0, 0.06]
# Modify the points to be more generalized (longer line)
A = [0, -0.01]
B = [0, 0.3]

df2018 = df2018.dropna()
df2018['Label'] = 0

for i in range(len(df2018)):
    if (0-df2018['mean'][i])*(0.3-df2018['std'][i])-(-0.01-df2018['std'][i])*(0-df2018['mean'][i]) > 0:
        df2018['Label'][i] = 'Red'
    elif (0-df2018['mean'][i])*(0.3-df2018['std'][i])-(-0.01-df2018['std'][i])*(0-df2018['mean'][i]) < 0:
        df2018['Label'][i] = 'Green'

plt.scatter(df2018['mean'], df2018['std'], c=df2018['Label'], s=15)
plt.xlabel('mean (μ)')
plt.ylabel('standard deviation (σ)')
plt.plot([0,0],[-0.01,0.3])

print('The plot shows that the line successfully assigns the label for year 2018.')
print('Red points locate on left side, green points locate on right side.')



# Question 3
# Implementing trading strategy: If it's Green, buy, if it's Red, sell
# Preprocessing the data
df2018_1 = df[df['Year'] == 2018]
df2018_1 = df2018_1[df2018_1['Weekday'] == 'Friday']
df2018_1 = df2018_1[['Week_Number', 'Weekday', 'Adj Close']]
df2018_1 = df2018_1.reset_index()
del df2018_1['index']
df2018_1['Label'] = df2018['Label']
df2018 = df2018_1

position = 0
share = 0
profit = 100
balance = [100,]
for i in range(len(df2018)-1):
    if df2017['Label'][i+1] == 'Green':
        if position == 0:
            share = profit/df2018['Adj Close'][i]
            profit = 0
            position = 1
    elif df2018['Label'][i+1] == 'Red':    
        if position != 0:
            profit = share * df2018['Adj Close'][i]
            share = 0
            position = 0
    balance.append(profit+share*df2018['Adj Close'][i])

end_value = balance[len(balance)-1]
print('The final value of the account for 2018 is $', round(end_value,2))







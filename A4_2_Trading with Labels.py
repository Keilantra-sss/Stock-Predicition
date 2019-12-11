#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 11:40:39 2019

@author: keilantra
"""

import os
import numpy as np 
import pandas as pd
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
    
# Select the year and useful columns
df = df[['Year', 'Weekday', 'Week_Number', 'Adj Close', 'Label']]
df2017 = df[df['Year'] == 2017]
df2017.index = range(len(df2017))
df2018 = df[df['Year'] == 2018]
df2018.index = range(len(df2018))

# For year 2017
position = 0
share = 0
profit = 100
balance = [100,]
for i in range(len(df2017)-1):
    if df2017['Weekday'][i] == 'Friday':
        # Assume making a decision on Friday
        # For the week with only Monday to Thursday, assume no trading 
        if df2017['Label'][i+1] == 'Green':
            if position == 0:
                share = profit/df2017['Adj Close'][i]
                profit = 0
                position = 1
        elif df2017['Label'][i+1] == 'Red':    
            if position != 0:
                profit = share * df2017['Adj Close'][i]
                share = 0
                position = 0
        balance.append(profit+share*df2017['Adj Close'][i])
        

# Question 1: Average and volatility of weekly balances
Mean = pd.DataFrame(balance).mean()
Std = pd.DataFrame(balance).std()
print('The average of weekly balances is',round(Mean[0],2))
print('The volatility of weekly balances is',round(Std[0],2))

# Question 2: Plot the growth of the account
x_axis = list(np.arange(0,51))
y_axis = balance   
plt.xlabel('Week Number')
plt.ylabel('Account Balance')
plt.plot(x_axis, y_axis)
plt.show()  

# Question 3: Min & Max of the account
Min = pd.DataFrame(balance).min()
Min_w = pd.DataFrame(balance).idxmin()
Max = pd.DataFrame(balance).max()
Max_w = pd.DataFrame(balance).idxmax()
print('The min of the account is',round(Min[0],2),'when week number is',Min_w[0])
print('The max of the account is',round(Max[0],2),'when week number is',Max_w[0])

# Question 4: Final value of the account
print('The final value of the account is', round(balance[50],2))

# Question 5:
balance = [round(i,2) for i in balance]
bal = pd.DataFrame(balance)
bal.columns = ['Balance']
# Calculate the growing periods
week_increase = []
increase = 0
i = 0
while i in range(len(bal)-1):
    if bal['Balance'][i+1] > bal['Balance'][i]:
        increase += 1
        i += 1
    else: 
        i += 1
        if increase != 0:
            week_increase.append(increase)
            increase = 0
            continue
print(week_increase)
print('The maximum duration that the account was growing is',max(week_increase),'weeks')
# Claculate the decreasing periods
week_decrease = []
decrease = 0
i = 0
while i in range(len(bal)-1):
    if bal['Balance'][i+1] < bal['Balance'][i]:
        decrease += 1
        i += 1
    else: 
        i += 1
        if decrease != 0:
            week_decrease.append(decrease)
            decrease = 0
            continue
print(week_decrease)
print('The maximum duration that the account was decreasing is',max(week_decrease),'weeks')


# For year 2018
position = 0
share = 0
profit = 100
balance = [100,]
for i in range(len(df2018)-1):
    if df2018['Weekday'][i] == 'Friday':
        # Assume making a decision on Friday
        # For the week with only Monday to Thursday, assume no trading
        if df2018['Label'][i+1] == 'Green':
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
        
# Question 1: Average and volatility of weekly balances
Mean = pd.DataFrame(balance).mean()
Std = pd.DataFrame(balance).std()
print('The average of weekly balances is',round(Mean[0],2))
print('The volatility of weekly balances is',round(Std[0],2))

# Question 2: Plot the growth of the account
x_axis = list(np.arange(0,52))
y_axis = balance   
plt.xlabel('Week Number')
plt.ylabel('Account Balance')
plt.plot(x_axis, y_axis)
plt.show()  

# Question 3: Min & Max of the account
Min = pd.DataFrame(balance).min()
Min_w = pd.DataFrame(balance).idxmin()
Max = pd.DataFrame(balance).max()
Max_w = pd.DataFrame(balance).idxmax()
print('The min of the account is',round(Min[0],2),'when week number is',Min_w[0])
print('The max of the account is',round(Max[0],2),'when week number is',Max_w[0])

# Question 4: Final value of the account
print('The final value of the account is', round(balance[51],2))

# Question 5:
balance = [round(i,2) for i in balance]
bal = pd.DataFrame(balance)
bal.columns = ['Balance']
# Calculate the growing periods
week_increase = []
increase = 0
i = 0
while i in range(len(bal)-1):
    if bal['Balance'][i+1] > bal['Balance'][i]:
        increase += 1
        i += 1
    else: 
        i += 1
        if increase != 0:
            week_increase.append(increase)
            increase = 0
            continue
print(week_increase)
print('The maximum duration that the account was growing is',max(week_increase),'weeks')
# Claculate the decreasing periods
week_decrease = []
decrease = 0
i = 0
while i in range(len(bal)-1):
    if bal['Balance'][i+1] < bal['Balance'][i]:
        decrease += 1
        i += 1
    else: 
        i += 1
        if decrease != 0:
            week_decrease.append(decrease)
            decrease = 0
            continue
print(week_decrease)
print('There is no duration of the decreaseing in value.')    

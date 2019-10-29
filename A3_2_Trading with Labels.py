#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 18:58:03 2019

@author: keilantra
"""

from pandas_datareader import data as web
import os
import math
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
    
    
# Get useful columns and add necessary columns
start_date='2014-01-01'; 
end_date='2018-12-31'
df = df[df['Date'] >= start_date]
df = df[df['Date'] <= end_date]
df = df[['Year', 'Day', 'Adj Close']]


# Calculate the result
result = pd.DataFrame(index = range(20,61,10),columns = range(5,26,5))
for L in range(20,61,10):  
    for S in range(5,26,5):
        profit = 0
        transaction = 0
        profit = 0
        position = 0
        if L > S:
            for x in range(L,len(df)):
                MA_L = np.mean(df.iloc[x-L:x,2]) # 2 refers to adj close
                MA_S = np.mean(df.iloc[x-S:x,2])
                if MA_L < MA_S and position == 0:
                    share = 100/df.iloc[x,2]
                    position = 1
                    price = df.iloc[x,2]
                elif MA_L > MA_S and position == 1:
                    profit += share * (df.iloc[x,2] - price)
                    position = 0
                    transaction += 1
            if transaction != 0:
                profit = profit / transaction
        else:
            profit = 0
        result.iloc[int((L-20)/10),int((S-5)/5)] = profit
                
# Plot the result
col = [] 
for L in range(20,61,10):  
    for S in range(5,26,5):
        if result.iloc[int((L-20)/10),int((S-5)/5)] > 0:
            col = 'green'
        elif result.iloc[int((L-20)/10),int((S-5)/5)] < 0:
            col = 'red'
        plt.scatter(L,S, s=100* abs(result.iloc[int((L-20)/10),int((S-5)/5)]), color = col)
plt.show()



















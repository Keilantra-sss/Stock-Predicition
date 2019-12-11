#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 17:39:44 2019

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
    
# Select the year
df2017 = df[df['Year'] == 2017]
df2018 = df[df['Year'] == 2018]


# For year 2017
# Calculate the μ and σ            
df2017 = df2017.groupby(['Week_Number','Label'])['Return'].agg([np.mean,np.std]) 
df2017 = df2017.reset_index()

# Plot the result
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


# For year 2018
# Calculate the μ and σ            
df2018 = df2018.groupby(['Week_Number','Label'])['Return'].agg([np.mean,np.std]) 
df2018 = df2018.reset_index()

# Plot the result
plt.figure(figsize=(13,13))
x2 = df2018['mean']
y2 = df2018['std']
pid2 = df2018['Week_Number']

for xi, yi, pidi in zip(x2,y2,pid2):
    plt.annotate(str(pidi), xy=(xi,yi))

plt.scatter(x2, y2, c=df2018['Label'],s=15)
plt.xlabel('mean (μ)')
plt.ylabel('standard deviation (σ)')
plt.show()


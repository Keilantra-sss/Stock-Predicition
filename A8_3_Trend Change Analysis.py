#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 17:39:21 2019

@author: keilantra
"""

import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import f as fisher_f

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


#for y in range(1,3):
def F_test(year):
    output_fs = []
    output_p = []
    candidate = []
    for m in range(1,13):        
#       dfy = df[df['Year'] == 2016+y]
        dfy = df[df['Year'] == year]
        dfm = dfy[dfy['Month'] == m]
        dfm = dfm.reset_index()
        
# Model 1
        X = dfm.loc[:,'index'].values.reshape(-1,1)
        Y = dfm.loc[:,'Adj Close'].values
        model = LinearRegression()
        model.fit(X,Y)
        y_pred = model.predict(X)
        sse = 0
        for i in range(len(dfm)):
            sse += (y_pred[i] - Y[i])**2

# Model 2&3
        result = []
        for k in range(1,len(dfm)-2):
            dfm_T1 = dfm.loc[:k,]
            dfm_T2 = dfm.loc[k+1:,]
    
            X = dfm_T1.loc[:,'index'].values.reshape(-1,1)
            Y = dfm_T1.loc[:,'Adj Close'].values
            model = LinearRegression()
            model.fit(X,Y)
            y_pred = model.predict(X)
            sse1 = 0
            for i in range(len(dfm_T1)):
                sse1 += (y_pred[i] - Y[i])**2
        
        
            X = dfm_T2.loc[:,'index'].values.reshape(-1,1)
            Y = dfm_T2.loc[:,'Adj Close'].values
            model = LinearRegression()
            model.fit(X,Y)
            y_pred = model.predict(X)
            sse2 = 0
            for i in range(len(dfm_T2)):
                sse2 += (y_pred[i] - Y[i])**2
        
            ssetotal = sse1 + sse2
            result.append(ssetotal)
        
        optimal = pd.DataFrame(result).idxmax()
        optimalk = optimal[0]+1
        candidate.append(optimalk+1) # for the reason that index begins with 0
        
        dfm_T1 = dfm.loc[:optimalk,]
        dfm_T2 = dfm.loc[optimalk+1:,]
    
        X = dfm_T1.loc[:,'index'].values.reshape(-1,1)
        Y = dfm_T1.loc[:,'Adj Close'].values
        model = LinearRegression()
        model.fit(X,Y)
        y_pred = model.predict(X)
        sse1 = 0
        for i in range(len(dfm_T1)):
            sse1 += (y_pred[i] - Y[i])**2
    
    
        X = dfm_T2.loc[:,'index'].values.reshape(-1,1)
        Y = dfm_T2.loc[:,'Adj Close'].values
        model = LinearRegression()
        model.fit(X,Y)
        y_pred = model.predict(X)
        sse2 = 0
        for i in range(len(dfm_T2)):
            sse2 += (y_pred[i] - Y[i])**2

        fs =((sse - sse1 - sse2)*(len(dfm)-4))/(2*(sse1 + sse2))
        output_fs.append(fs)
        p = fisher_f.cdf(fs,2,16)
        output_p.append(p)
        c = {'day':candidate, 'fs':output_fs, 'p':output_p}
        output = pd.DataFrame(c)
    return output


# Calculate the p-value for year 2017 and year 2018
p2017 = F_test(2017)
p2018 = F_test(2018)        
        
# Question 1
# For 2017
for i in range(len(p2017)):
    print('For month',i+1,'the candidate days is',p2017['day'][i])

count = 0
for i in range(len(p2017)):
    if p2017['p'][i] > 0.1:
        count += 1
        print('For month',i+1,'there is a significant change of pricing trend.')
    else:
        print('For month',i+1,'there is no significant change of pricing trend.')
        
# For 2018
for i in range(len(p2018)):
    print('For month',i+1,'the candidate days is',p2018['day'][i])
    
count2 = 0
for i in range(len(p2018)):
    if p2018['p'][i] > 0.1:
        count2 += 1
        print('For month',i+1,'there is a significant change of pricing trend.')
    else:
        print('For month',i+1,'there is no significant change of pricing trend.')              

        
# Question 2
print('There are',count,'months exhibit significant price change for year 2017.')
print('There are',count2,'months exhibit significant price change for year 2018.')


# Question 3
if count > count2:
    print('There are more changes in year 2017.')
else:
    print('There are more changes in year 2018.')

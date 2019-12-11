#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 12:45:42 2019

@author: keilantra
"""


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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

#Get useful columns and add necessary columns
df1 = df[['Year', 'Day', 'Adj Close']]
col_name = df1.columns.tolist()
col_name.insert(3,'Predict')
col_name.insert(4,'Label')
df1= df1.reindex(columns = col_name)

#Get the data of year 2017 and year 2018
df2017 = df1[df1['Year'] == 2017]
df2018 = df1[df1['Year'] == 2018]
df2017.index = range(len(df2017))
df2018.index = range(len(df2018))


# Linear regression 
lin_reg = LinearRegression(fit_intercept=True)

# Analyze year 2017
W = np.arange(5,31,1)
profit = []
for i in W:    
    position = 0
    long = 0
    short = 0
    longvalue = 0
    shortvalue = 0
    transaction = 0
    for j in range(len(df2017)-i):      
        y = df2017.loc[j:j+i-1,'Adj Close']
        x = np.array(range(1,i+1,1))
        x = x[:,np.newaxis]
        lin_reg.fit(x,y)        
        y_pred = lin_reg.predict(np.array([[i+1]]))
        P = df2017.loc[j+i-1,'Adj Close']
        if y_pred > P:
            if position == 0:
                longvalue = P
                share = 100/longvalue
                position = -1     # -1 stands for long position             
            elif position == 1:    # 1 stands for short position
                short += share * (shortvalue - P)
                share = 0
                position = 0
                transaction += 1                
        elif y_pred < P:
            if position == 0:
                shortvalue = P
                share = 100/shortvalue
                position = 1               
            elif position == -1:
                long += share * (P - longvalue)
                share = 0
                position = 0
                transaction += 1
    if transaction != 0:
        profit.append((long + short)/transaction)

# Get the profit for each W      
result = pd.DataFrame(profit, W)


# Question 1
# Plot of P/L per trade for each W            
x_axis = W
y_axis = profit
plt.scatter(x_axis,y_axis)
plt.show()       
        
# Get the W star here 
optimal = result.idxmax()           
print('The optimal W is:',optimal[0])



# Analyze year 2018
i = optimal[0]
r2 = []

position = 0
long, short = 0, 0
l_count, s_count = 0, 0
longvalue, shortvalue = 0, 0
longday = 0
shortday = 0
longpoint = 0
shortpoint = 0 

for j in range(len(df2018)-i):  
    y = df2018.loc[j:j+i-1,'Adj Close']
    x = np.array(range(1,1+i,1))
    x = x[:,np.newaxis]
    lin_reg.fit(x,y)
    r2.append(lin_reg.score(x, y))
       
    y_pred = lin_reg.predict(np.array([[i+1]]))
    P = df2018.loc[j+i-1,'Adj Close']
    if y_pred > P:
        if position == 0:
            longvalue = P
            share = 100/longvalue
            longpoint = j
            position = -1    # -1 stands for long position              
        elif position == 1:  # 1 stands for short position
            short += share * (shortvalue - P)
            shortday = shortday + j - shortpoint
            share = 0
            position = 0
            s_count += 1         
    elif y_pred < P:
        if position == 0:
            shortvalue = P
            shortpoint = j
            share = 100/shortvalue
            position = 1         
        elif position == -1:
            long += share * (P - longvalue)
            longday = longday + j - longpoint
            share = 0
            position = 0
            l_count +=1


# Question 2
# Plot the r2 for year 2018
x_axis = np.arange(i,len(df2018), 1)
y_axis = r2
plt.plot(x_axis, y_axis)
plt.show()

# Calculate the average r2
r_avg = sum(r2)/len(r2)
print('The average r2 is',round(r_avg,3))
print('The r2 is less than 0.5, which means the model did not explain the price well.')

# Question 3: Count the long_position and the short_position
print('There are',l_count,'long position transactions.')
print('There are',s_count,'short position transactions.')

# Queston 4: Average P/L 
print('The average P/L per long position trade is $',round(long/l_count,2))
print('The average P/L per short position trade is $',round(short/s_count,2))

# Question 5: Average number of days
print('The average number of days per long position trade is',round(longday/l_count,2))
print('The average number of days per short position trade is',round(shortday/s_count,2))


# Question 6
# Perform the overall year 2018 analysis on year 2017
i = optimal[0]
r2 = []

position = 0
long, short = 0, 0
l_count, s_count = 0, 0
longvalue, shortvalue = 0, 0
longday = 0
shortday = 0
longpoint = 0
shortpoint = 0 

for j in range(len(df2017)-i):  
    y = df2017.loc[j:j+i-1,'Adj Close']
    x = np.array(range(1,1+i,1))
    x = x[:,np.newaxis]
    lin_reg.fit(x,y)
    r2.append(lin_reg.score(x, y))
       
    y_pred = lin_reg.predict(np.array([[i+1]]))
    P = df2017.loc[j+i-1,'Adj Close']
    if y_pred > P:
        if position == 0:
            longvalue = P
            share = 100/longvalue
            longpoint = j
            position = -1    # -1 stands for long position              
        elif position == 1:  # 1 stands for short position
            short += share * (shortvalue - P)
            shortday = shortday + j - shortpoint
            share = 0
            position = 0
            s_count += 1         
    elif y_pred < P:
        if position == 0:
            shortvalue = P
            shortpoint = j
            share = 100/shortvalue
            position = 1         
        elif position == -1:
            long += share * (P - longvalue)
            longday = longday + j - longpoint
            share = 0
            position = 0
            l_count +=1


# Plot the r2 for year 2017
x_axis = np.arange(i,len(df2017), 1)
y_axis = r2
plt.plot(x_axis, y_axis)
plt.show()

# Calculate the average r2
r_avg = sum(r2)/len(r2)
print('The average r2 is',round(r_avg,3))
print('The r2 is slightly greater than 0.5, which means the model can explain the price at an average level.')

# Count the long_position and the short_position
print('There are',l_count,'long position transactions.')
print('There are',s_count,'short position transactions.')

# Average P/L 
print('The average P/L per long position trade is $',round(long/l_count,2))
print('The average P/L per short position trade is $',round(short/s_count,2))

# Average number of days
print('The average number of days per long position trade is',round(longday/l_count,2))
print('The average number of days per short position trade is',round(shortday/s_count,2))

print('There are differences between year 2017 and year 2018, but the differences are significant.')

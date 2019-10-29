#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 15:28:20 2019

@author: keilantra
"""

#First: run file stock_data.py to get the data

#Select the year of data
start_date='2017-01-01'
end_date='2018-12-31'
df = df[df['Date'] >= start_date]
df = df[df['Date'] <= end_date]

#Perform the "Inertia" strategy
share_long = 0
share_shrot = 0
profit_long = 0
profit_short = 0

for i in range(len(df)):
    if df['Open'][i+1] > df['Adj Close'][i]:
        share_long = 100/df['Open'][i+1]
        profit_long += share_long * (df['Adj Close'][i+1] - df['Open'][i+1])
    elif df['Open'][i+1] < df['Adj Close'][i]:
        share_short = 100/df['Open'][i+1]
        profit_short += share_short * (df['Open'][i+1] - df['Adj Close'][i+1]) 


profit_long = float(format(profit_long, '0.2f'))
profit_short = float(format(profit_short, '0.2f'))

profit_year = profit_long + profit_short
profit_avg = float(format(profit_year/(len(df)-1), '0.2f'))

#Question 1
print('The average daily profit is $', profit_avg)

#Question 2
print('The profit from long position is $', profit_long)
print('The profit from short position is $', profit_short)

if profit_long > profit_short:
    print('The profit from “long” position is higher than the profit from “short” position')
    print('long position is more profitable.')
else:
    print('The profit from “long” position is lower than the profit from “short” position')
    print('Short position is more profitable.')




#Add a restriction
import numpy as np 
import matplotlib.pyplot as plt    
import os

xprofit = []
threshold = []

for j in range(0,101):
    x = j/1000
    threshold.append(x)
    profit_total = []
    trade = 0
    for i in range(len(df)-1):
        return_rate = (df['Open'][i+1] - df['Adj Close'][i])/df['Adj Close'][i]
        if abs(return_rate) > x:
            if df['Open'][i+1] > df['Adj Close'][i]:
                trade += 1
                profit_total.append(100/df['Open'][i+1] * (df['Adj Close'][i+1]- df['Open'][i+1]))
            elif df['Open'][i+1] < df['Adj Close'][i]:
                trade += 1
                profit_total.append(100/df['Open'][i+1] * (df['Open'][i+1]- df['Adj Close'][i+1]))
    xprofit.append(sum(profit_total)/trade)
    

#Question 3   
#Plot the average profit    
plot_dir = r'/Users/keilantra/Desktop/'
x_axis = threshold
y_axis = xprofit
fig = plt.figure()
plt.plot(x_axis, y_axis)
plt.title('Average Profit per Trade',loc='center')
output_file = os.path.join('Average Profit per Trade.pdf')
plt.show()
fig.savefig(output_file)

#Optimal values for x
optimal = max(enumerate(xprofit),key=lambda x:x[1])[0]         
print('The optimal values for x is', x_axis[optimal])

#Discuss the findings
print('When 0<x<6%, the average profit per trade shows a general increasing trend while x increases;')
print('When 6%<x<10%, the average profit per trade shows a general decreasing trend while x increases.')



#Question 4
xprofit_long = []
xprofit_short =[]
threshold = []

for j in range(0,101):
    x = j/1000
    threshold.append(x)  
    profit_long = []
    profit_short = []
    trade_long = 0
    trade_short = 0
    for i in range(len(df)-1):
        return_rate = (df['Open'][i+1] - df['Adj Close'][i])/df['Adj Close'][i]
        if abs(return_rate) > x:
            if df['Open'][i+1] > df['Adj Close'][i]:
                trade_long += 1
                profit_long.append(100/df['Open'][i+1] * (df['Adj Close'][i+1]- df['Open'][i+1]))
            elif df['Open'][i+1] < df['Adj Close'][i]:
                trade_short += 1
                profit_short.append(100/df['Open'][i+1] * (df['Open'][i+1]- df['Adj Close'][i+1]))
    xprofit_long.append(sum(profit_long)/trade_long)
    xprofit_short.append(sum(profit_short)/trade_short)

#Plot the average profit under long position      
plot_dir = r'/Users/keilantra/Desktop/'
x_axis = threshold
y_axis = xprofit_long
fig = plt.figure()
plt.plot(x_axis, y_axis)
plt.title('Average Profit per Trade (Long Position)',loc='center')
output_file = os.path.join('Average Profit per Trade (Long Position).pdf')
plt.show()
fig.savefig(output_file)

#Optimal values for x (long position)
optimal_long = max(enumerate(xprofit_long),key=lambda x:x[1])[0]         
print('The optimal values for x under long position is', x_axis[optimal_long])


#Plot the average profit under short position      
plot_dir = r'/Users/keilantra/Desktop/'
x_axis = threshold
y_axis = xprofit_short
fig = plt.figure()
plt.plot(x_axis, y_axis)
plt.title('Average Profit per Trade (Short Position)',loc='right')
output_file = os.path.join('Average Profit per Trade (Short Position).pdf')
plt.show()
fig.savefig(output_file)

#Optimal values for x (long position)
optimal_short = max(enumerate(xprofit_short),key=lambda x:x[1])[0]         
print('The optimal values for x under short position is', x_axis[optimal_short])

#Discuss the findings
print('The average profit per trade under long position shows a general decreasing trend while x increases. The profits are generally negative.')
print('The pattern of the average profit per trade under short position is similar to the pattern of the overall average profit per trade. The profits are mostly positive and slightly higher than the overall average profit per trade.')
print('In conclusion, taking the short position strategy would be a better choice for this stock.')


#Assignment 2
#lable "Green/Red" results are in the excel file
#The rule: 
#If the Adj Close price this Friday is higher than last Friday, it is green, otherwise red.
#If Friday is not open, then taking Thursday as the last day of the week, and so on.
#In general, if the Adj Close price of the last day of the week is higher than the last day of the last week, it is green, otherwise red.


       



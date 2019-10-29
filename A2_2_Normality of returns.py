#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 10:45:16 2019

@author: keilantra
"""

from pandas_datareader import data as web
import os
import math
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

#Import the data
# run this  !pip install pandas_datareader

def get_stock(ticker, start_date, end_date, s_window, l_window):
    try:
        df = web.get_data_yahoo(ticker, start=start_date, end=end_date)
        df['Return'] = df['Adj Close'].pct_change()
        df['Return'].fillna(0, inplace = True)
        df['Date'] = df.index
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year 
        df['Day'] = df['Date'].dt.day
        for col in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
            df[col] = df[col].round(2)
        df['Weekday'] = df['Date'].dt.weekday_name  
        df['Week_Number'] = df['Date'].dt.strftime('%U')
        df['Year_Week'] = df['Date'].dt.strftime('%Y-%U')
        df['Short_MA'] = df['Adj Close'].rolling(window=s_window, min_periods=1).mean()
        df['Long_MA'] = df['Adj Close'].rolling(window=l_window, min_periods=1).mean()        
        col_list = ['Date', 'Year', 'Month', 'Day', 'Weekday', 
                    'Week_Number', 'Year_Week', 'Open', 
                    'High', 'Low', 'Close', 'Volume', 'Adj Close',
                    'Return', 'Short_MA', 'Long_MA']
        df = df[col_list]
        return df
    except Exception as error:
        print(error)
        return None

def get_last_digit(y):
        x = str(round(float(y),2))
        x_list = x.split('.')
        fraction_str = x_list[1]
        if len(fraction_str)==1:
            return 0
        else:
            return int(fraction_str[1])


ticker='PCG'
start_date='2014-01-01'
end_date='2018-12-31'
s_window = 14
l_window = 50
input_dir = r'/Users/keilantra/Desktop/'
output_file = os.path.join(input_dir, ticker + '.csv')

df = get_stock(ticker, start_date, end_date, s_window, l_window)

df.to_csv(output_file, index=False)



df['last digit'] = df['Open'].apply(get_last_digit)

df['count'] = 1
total = len(df)

df_1 = df.groupby(['last digit'])['count'].sum()
df_2 = df_1.to_frame()
df_2.reset_index(level=0, inplace=True)
df_2['digit_frequency'] = df_2['count']/total
df_2['uniform'] = 0.10

output_file = os.path.join(input_dir, ticker + '_digit_analysis.csv')
df_2.to_csv(output_file, index=False)


#df.to_csv(output_file, index=False)
#with open(output_file) as f:
#    lines = f.read().splitlines()
#df[['Short_MA', 'Long_MA', 'Adj Close']].plot()
#df_2 = pd.read_csv(output_file)

    
#Select the year of data, change the year here
year = '2018'
start_date = year + '-01-01'
end_date = year + '-12-31'
df = df[df['Date'] >= start_date]
df = df[df['Date'] <= end_date]



#Compute the positive and negative return
tradeday = len(df)
pos = len(df[df['Return'] > 0])
neg = len(df[df['Return'] < 0])

#Question 1: Answer
print('In year', year,':')
print('There are', pos, 'days with positive returns.')
print('There are', neg, 'days with negative returns.')



#Compute the average daily returns μ
return_avg = df['Return'].mean()
return_upper = len(df[df['Return'] > return_avg])/tradeday
return_lower = len(df[df['Return'] < return_avg])/tradeday

#Question 2: Answer
if pos > neg:
    print('In year',year, ':')
    print('There are', tradeday, 'trading days.')
    print('There are more positive return days.')
else:
    print('In year',year, ':')
    print('There are', tradeday, 'trading days.')
    print('There are more negative return days.')
    
print('The mean (μ) is',round(return_avg,3))
print('%.2f%%'%(round((100*return_upper),2)),'of return is greater than μ.')
print('%.2f%%'%(round((100*return_lower),2)),'of return is less than μ.')


    
#Compute the mean and stdev
return_avg = df['Return'].mean()
return_std = df['Return'].std()
rbound = return_avg + 2 * return_std
lbound = return_avg - 2 * return_std
return_r = len(df[df['Return'] > rbound])/tradeday
return_l = len(df[df['Return'] < lbound])/tradeday

#Question 3: Answer
print('In year', year,':')
print('There are', tradeday, 'trading days.')
print('The mean (μ) is',round(return_avg,3))
print('The standard deviation (σ) is',round(return_std,3))
print('%.3f%%'%(round(100*return_r,3)),'of return is larger than mean + 2std.')
print('%.3f%%'%(round(100*return_l,3)),'of return is smaller than mean - 2std.')

#The result table is shown in the word file.


# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 15:28:20 2019

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


#Compute the distribution of the last digit
format_round = lambda x: '%.2f' % x
df['Open'] = df['Open'].map(format_round)

format_digit = lambda x: str(x)[4]
df['Last_digit'] = df['Open'].map(format_digit)

df['Frequency'] = 1

digit_count = df.groupby(df['Last_digit'])['Frequency'].sum()
actual = digit_count/len(df)

#Question 1: Get the most frequent digit
most_digit = [i for i, x in enumerate(actual) if x == np.max(actual)]
print('The most frequent digit is', most_digit, 'with the frequency of', max(actual))

#Question 2: Get the least frequent digit
least_digit = [i for i, x in enumerate(actual) if x == np.min(actual)]
print('The least frequent digit is', least_digit, 'with the frequency of', min(actual))

#Calculate the error
target = np.array(actual)
prediction = np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])

error = []
for i in range (len(target)):
    error.append(actual[i] - prediction[i])

sqError = []
absError = []
for i in error:
    sqError.append(i*i)
    absError.append(abs(i))
    
max_absError = max(absError)
median_absError = np.median(absError)
mean_absError = np.mean(absError)
RMSE = np.sqrt(np.mean(absError))

#Question 3
print('In the year', year)
print('The max absolute error is', max_absError.round(3))
print('The median absolute error is', median_absError.round(3))
print('The mean absolute error is', mean_absError.round(3))
print('The root mean squared error (RMSE) is', RMSE.round(3))

#The result table is shown in the word file.

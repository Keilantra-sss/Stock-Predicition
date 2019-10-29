#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 18:58:03 2019

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
   
    
#Select the year of data
start_date='2017-01-01'; 
end_date='2018-12-31'
df = df[df['Date'] >= start_date]
df = df[df['Date'] <= end_date]

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

#Analyze year 2017
result = pd.DataFrame(index = range(10,51),columns = [0.5,1,1.5,2,2.5])
for i in range(10,51,1):   # i stands for W
    for k in [0.5,1,1.5,2,2.5]:   # k stands for k
        position = 0
        long = 0
        short = 0
        longvalue = 0
        shortvalue = 0
        transaction = 0
        pro_avg = 0
        for j in range(len(df2017)-i):
            P = df2017.loc[j+i-1,'Adj Close']
            MA = np.mean(df2017.loc[j:j+i,'Adj Close'])
            std = np.std(df2017.loc[j:j+i,'Adj Close'])
            upper = MA + k * std
            lower = MA - k * std
            if P > upper:
                if position == 0:
                    longvalue = P
                    share = 100 / longvalue
                    position = 1      #1 stands for short position
                elif position == -1:  #-1 stands for long position
                    short += share * (shortvalue - P)
                    share = 0
                    position = 0
                    transaction += 1               
            elif P < lower:
                if position == 0:
                    shortvalue = P
                    share = 100 / shortvalue
                    position = -1    #-1 stands for long position            
                elif position == 1:  #1 stands for short position
                    long += share * (P - longvalue)
                    share = 0
                    position = 0
                    transaction +=1
        if transaction != 0:
            pro_avg = (long + short)/transaction       
        result.iloc[i-10,int(k*2)-1] = pro_avg

#Plot the result
for i in range(10,51):
    for j in range(5):
        if result.iloc[i-10,j] > 0:
            col = 'green'
        elif result.iloc[i-10,j] < 0:
            col = 'red'
        plt.scatter(i,int(j+1)/2,s=abs(result.iloc[i-10,j]),color = col)
plt.show()

#Find the best combination for year 2017
result.max()
best = result[result == result.max()[1.0]]
print('The best combination of W and k is: W=17, k=1.0')


#Analyze year 2018
result = pd.DataFrame(index = range(10,51),columns = [0.5,1,1.5,2,2.5])
for i in range(10,51,1):   # i stands for W
    for k in [0.5,1,1.5,2,2.5]:   # k stands for k
        position = 0
        long = 0
        short = 0
        transaction = 0
        pro_avg = 0
        for j in range(len(df2018)-i):
            P = df2018.loc[j+i-1,'Adj Close']
            MA = np.mean(df2018.loc[j:j+i,'Adj Close'])
            std = np.std(df2018.loc[j:j+i,'Adj Close'])
            upper = MA + k * std
            lower = MA - k * std
            if P > upper:
                if position == 0:
                    longvalue = P
                    share = 100 / longvalue
                    position = 1      #1 stands for short position
                elif position == -1:  #-1 stands for long position
                    short += share * (shortvalue - P)
                    share = 0
                    position = 0
                    transaction += 1               
            elif P < lower:
                if position == 0:
                    shortvalue = P
                    share = 100 / shortvalue
                    position = -1    #-1 stands for long position            
                elif position == 1:  #1 stands for short position
                    long += share * (P - longvalue)
                    share = 0
                    position = 0
                    transaction +=1
        if transaction != 0:
            pro_avg = (long + short)/transaction       
        result.iloc[i-10,int(k*2)-1] = pro_avg

#Plot the result
for i in range(10,51):
    for j in range(5):
        if result.iloc[i-10,j] > 0:
            col = 'green'
        elif result.iloc[i-10,j] < 0:
            col = 'red'
        plt.scatter(i,int(j+1)/2,s=abs(result.iloc[i-10,j]),color = col)
plt.show()

#Find the best combination for year 2018
result.max()
best = result[result == result.max()[2.5]]
print('The best combination of W and k is: W=10,11,12,13,15,33, k=2.5')






















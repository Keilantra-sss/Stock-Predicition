#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 21:22:53 2019

@author: keilantra
"""

import numpy as np
import pandas as pd


# Import data
try:   
    df = pd.read_csv(r'/Users/keilantra/Desktop/tips.csv')
except Exception as e:
    print(e)
    print('failed to read the data') 
    
    
# Question 1: Average tip for lunch and for dinner
df['percentage'] = df['tip']/df['total_bill']*100
dfq1 = df.groupby('time')['percentage'].mean()
print('The average tip for lunch is','%.2f%%' % dfq1.loc['Lunch'])
print('The average tip for dinner is','%.2f%%' % dfq1.loc['Dinner'])


# Question 2: Average tip for each day of the week
dfq2 = df.groupby('day')['percentage'].mean()
print('The average tip for Thursday is','%.2f%%' % dfq2.loc['Thur'])
print('The average tip for Friday is','%.2f%%' % dfq2.loc['Fri'])
print('The average tip for Saturday is','%.2f%%' % dfq2.loc['Sat'])
print('The average tip for Sunday is','%.2f%%' % dfq2.loc['Sun'])


# Question 3: highest tips
dfq3 = df.groupby(['time','day'])['percentage'].mean()
#print('The time has the highest tip is',dfq1.idxmax())
#print('The day has the highest tip is',dfq2.idxmax() )
print('The day and time has the highest tips are',dfq3.idxmax())


# Question 4: Correlation between meal price and tips
cor = np.corrcoef(df['total_bill'],df['percentage'])[0][1]
print('The correlation between meal price and average tip is:', round(cor,2))
#cor2 = np.corrcoef(df['total_bill'],df['tip'])[0][1]
#print('The correlation between meal price and tip amount is:', round(cor2,2))


# Question 5: Relationships between tips and size
cor3 = np.corrcoef(df['tip'],df['size'])[0][1]
print('The correlation between meal price and tip amount is:', round(cor3,2))
cor4 = np.corrcoef(df['percentage'],df['size'])[0][1]
print('The correlation between meal price and tip percentage is:', round(cor4,2))
print('The larger the size of the group, the higher the tips, but the tip percentage goes down.')


# Question 6: Smoking people
dfq6 = df[df['smoker'] == 'Yes']
smoker = len(dfq6)/len(df)*100
print('There are','%.2f%%' % smoker,'of people are smoking')


# Question 7: Are tips increasing
time = list(range(len(df)))
cor4 = np.corrcoef(df['percentage'],time)[0][1]

if cor4 > 0:
    print('Tips are increasing with time in each day.')
elif cor4 < 0:
    print('Tips are decreasing with time in each day.')
else:
    print('There is no relationships between tips and time.')
    
print('The correlation between tips and time is',round(cor4,4))


# Question 8: correlation for smokers and non-smokers
smoker_tip = df[df['smoker'] == 'Yes']['percentage'].mean()
non_smoker_tip = df[df['smoker'] == 'No']['percentage'].mean()
print('The average tip for smokers is','%.2f%%' % round(smoker_tip,2)) 
print('The average tip for non-smokers is','%.2f%%' % round(non_smoker_tip,2)) 

if smoker_tip > non_smoker_tip:
    print('Smokers may give more tips.')
elif smoker_tip < non_smoker_tip:
    print('Non-smokers may give more tips.')
else:
    print('There is no difference between smokers and non-smokers.')

if abs(smoker_tip - non_smoker_tip) < 1:
    print('The difference between smokers and non-smokers is small.')
else:
    print('The difference between smokers and non-smokers is significant.')

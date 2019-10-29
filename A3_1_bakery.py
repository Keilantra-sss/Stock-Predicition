#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 18:11:35 2019

@author: keilantra
"""

import os
import pandas as pd
import math
import numpy as np 
import random

# First run 'transaction_from_bakery.py' to get data

# Question 1
df_hr = df.groupby(['Hour'])['Transaction'].nunique()
df_weekday = df.groupby(['Weekday'])['Transaction'].nunique()
df_period = df.groupby(['Period'])['Transaction'].nunique()
busiest_hr = df_hr.idxmax()
busiest_weekday = df_weekday.idxmax()
busiest_period = df_period.idxmax()

print('In terms of transactions:')
print('The busiest hour is',busiest_hr,'with',df_hr.max(),'transactions')
print('The busiest day of the week is',busiest_weekday,'with',df_weekday.max(),'transactions')
print('The busiest period is',busiest_period,'with',df_period.max(),'transactions')


# Question 2
df_hr = df.groupby(['Hour'])['Item_Price'].sum()
df_weekday = df.groupby(['Weekday'])['Item_Price'].sum()
df_period = df.groupby(['Period'])['Item_Price'].sum()
busiest_hr = df_hr.idxmax()
busiest_weekday = df_weekday.idxmax()
busiest_period = df_period.idxmax()

print('In terms of revenue:')
print('The busiest hour is',busiest_hr)
print('The busiest day of the week is',busiest_weekday)
print('The busiest period is',busiest_period)


# Question 3
df_item = df.groupby(['Item'])['Transaction'].nunique()
df_item1 = df_item.to_dict()
def get_keys(d, value):
    return [k for k,v in d.items() if v == value]

print('The most popular item is',get_keys(df_item1, df_item.max()))
print('The least popular item is',get_keys(df_item1, df_item.min()))


# Question 4
# For each day of the week
df1 = df
def get_weekday_barristas(weekday):
    df1 = df[df['Weekday'] == weekday]
    df1 = df1.groupby(['Year','Month','Day'])['Transaction'].nunique()
    df1 = df1.sort_values(ascending = False)
    barristas = math.ceil(df1.max()/50)
    print('We need',barristas,'barristas for', weekday)

get_weekday_barristas('Monday')    
get_weekday_barristas('Tuesday')
get_weekday_barristas('Wednesday')
get_weekday_barristas('Thursday')
get_weekday_barristas('Friday')
get_weekday_barristas('Saturday')
get_weekday_barristas('Sunday')

# Find the maximum barristas per day
df_day = df.groupby(['Weekday','Year','Month','Day'])['Transaction'].nunique()
barristas_all = math.ceil(df_day.max()/50)
print('We need',barristas_all,'barristas for each day of the week.')


# Question 5
df['Group'] = 'None'
for i in range(len(df)):
    if df.iloc[i,9] in ('Coffee','Tea','Hot chocolate','Juice',\
                         'Coke','Mineral water','Smoothies'):
        df.iloc[i,11] = 'Drinks'
    elif df.iloc[i,9] in ('Alfajores','Bacon','Baguette','Bare Popcorn',\
                'Bread','Bread Pudding','Brioche and salami','Brownie',\
                'Cake','Caramel bites','Cherry me Dried fruit','Chicken Stew',\
                'Chicken sand','Chimichurri Oil','Chocolates','Cookies',\
                'Crepes'  ,'Crisps','Duck egg','Dulce de Leche','Eggs',\
                'Empanadas','Extra Salami or Feta','Focaccia','Frittata',\
                'Fudge','Gingerbread syrup','Granola', 'Honey','Jam',\
                'Kids biscuit','Lemon and coconut','Medialuna',\
                'Mighty Protein','Muesli','Muffin','Olum & polenta',\
                'Pastry','Polenta','Raspberry shortbread sandwich',\
                'Salad','Sandwich','Scone','Soup','Spanish Brunch',\
                'Tacos/Fajita','Tartine','Tiffin','Toast','Truffles'\
                ,'Vegan Feast','Vegan mincepie'):
        df.iloc[i,11] = 'Food' 
    else:
        df.iloc[i,11] = 'Unknown'

df_food = df[df['Group'] == 'Food']
df_drink = df[df['Group'] == 'Drinks']

food_avg = df_food['Item_Price'].mean()
drink_avg = df_drink['Item_Price'].mean()

print('The average price of a drink item is','%.2f' % drink_avg)
print('The average price of a food item is','%.2f' % food_avg)


# Question 6
if drink_avg > food_avg:
    print('The coffee shop makes more money from selling drinks.')
else:
    print('The coffee shop makes more money from selling food.')


# Question 7 & Question 8
df2 = df
def get_weekday_item(weekday):
    df2 = df[df['Weekday'] == weekday]
    df2 = df2.groupby(['Item'])['Transaction'].nunique()
    df2 = df2.sort_values(ascending = False)
    print('The top 5 most popular items for ' + weekday)
    print(df2.head(5))
    print('--------------------------------------------')
    print('The bottom 5 least popular items for ' + weekday)
    print(df2.tail(5))

get_weekday_item('Monday')    
get_weekday_item('Tuesday')
get_weekday_item('Wednesday')
get_weekday_item('Thursday')
get_weekday_item('Friday')
get_weekday_item('Saturday')
get_weekday_item('Sunday')

print('The lists do not stay the same from day to day')


# Question 9
drink_per = df_drink['Transaction'].nunique()/df['Transaction'].max()
print('There are','%.2f'% drink_per,'drinks per transaction')





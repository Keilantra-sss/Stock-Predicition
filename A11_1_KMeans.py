#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 17:50:03 2019

@author: keilantra
"""

import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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
    
# Data preparation
start_date='2017-01-01'
df = df[df['Date'] >= start_date]
df = df.reset_index()

df2 = df[['Year','Year_Week','Adj Close','Label']]
df2 = df2.sort_values(by = ['Year_Week'])
df2 = df2[:-1]

df2 = df2.groupby(['Year_Week','Label'])['Adj Close'].agg([np.mean,np.std]) 
df2 = df2.reset_index()

# Preprocessing data
feature = ['mean','std']
X = df2[feature].values
X = StandardScaler().fit_transform(X)


# Question 1
# Perform k-means (k=3)
kmeans_classifier = KMeans(n_clusters=3)
y_kmeans = kmeans_classifier.fit_predict(X)
centroids = kmeans_classifier.cluster_centers_

# Plot the cluster
colmap = ['red','green','blue','yellow','black']
df2['Cluster'] = y_kmeans
for i in range (3):
    new_df = df2[df2['Cluster'] == i]
    plt.scatter(new_df['mean'], new_df['std'], color=colmap[i], s=30)
plt.show()

# Fine the best k
distortion = []
for k in range(1,9):
    kmeans_classifier = KMeans(n_clusters=k)
    y_kmeans = kmeans_classifier.fit_predict(X)
    inertia = kmeans_classifier.inertia_
    distortion.append(inertia)
    
# Plot the knee (elbow chart)
plt.plot(range(1, 9), distortion, marker='o')
plt.xlabel('number of clusters: k')
plt.ylabel('distortion')
plt.show()
print('The knee method shows that the best k is 3')

    
# Question 2
kmeans_classifier = KMeans(n_clusters=3)
y_kmeans = kmeans_classifier.fit_predict(X)
centroids = kmeans_classifier.cluster_centers_
df2['Cluster'] = y_kmeans

# Plot the cluster
for i in range (3):
    new_df = df2[df2['Cluster'] == i]
    plt.scatter(new_df['mean'], new_df['std'], color=colmap[i], s=30 )
plt.show()

# Compute the percentage
Green = []
Red = []  
for i in range(3):
    new_df = df2[df2['Cluster'] == i]
    green = new_df['Label'][new_df['Label'] == 'Green'].count()/len(new_df)
    red = new_df['Label'][new_df['Label'] == 'Red'].count()/len(new_df)
    Green.append(green)
    Red.append(red)
    print('For cluster',i)
    print('The percentage of green is: %.2f%%' % (green*100))
    print('The percentage of red is: %.2f%%' % (red*100))
    
    
# Question 3  
for i in range(3):
    if Green[i] > 0.9:
        print('The k-means clustering finds pure clusters, and the pure cluster is',i)
    elif Green[i] < 0.1:
        print('The k-means clustering finds pure clusters, and the pure cluster is',i)
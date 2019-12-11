#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 09:13:36 2019

@author: keilantra
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import recall_score
from sklearn import metrics

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

# The 15th week in 2017 and 12nd week in 2018 have no Friday
# Insert the Thursday data to the dataframe
df2 = df[df['Weekday'] == 'Friday']
df2.loc[1] = df.iloc[70,]
df2.loc[2] = df.iloc[311,]
df2 = df2.sort_values(by = ['Year','Month','Day'])
df2_17 = df2[df2['Year'] == 2017].reset_index()
df2_18 = df2[df2['Year'] == 2018].reset_index()

# Calculate the μ and σ for year 2017 
df2017 = df[df['Year'] == 2017]
df2017 = df2017.groupby(['Week_Number','Label'])['Adj Close'].agg([np.mean,np.std])
df2017 = df2017.reset_index()
# Insert the Adj Close to year 2017
df2017['Adj Close'] = df2_17['Adj Close'] 
del df2_17

# Calculate the μ and σ for year 2018
df2018 = df[df['Year'] == 2018]
df2018 = df2018.groupby(['Week_Number','Label'])['Adj Close'].agg([np.mean,np.std])
df2018 = df2018.reset_index()
# Insert the Adj Close to year 2018
df2018['Adj Close'] = df2_18['Adj Close'] 
del df2_18


# Preprocessing data
attribute = ['mean','std']
X = df2017[attribute].values
df2017['Label'][df2017['Label'] == 'Green'] = 1
df2017['Label'][df2017['Label'] == 'Red'] = 0
y = df2017.loc[:, 'Label'].values  
le = LabelEncoder()
X = StandardScaler().fit_transform(X)
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=369)
# For y: 1 refers to green, 0 refers to red
X2 = df2018[attribute].values
le = LabelEncoder()
df2018['Label'][df2018['Label'] == 'Green'] = 1
df2018['Label'][df2018['Label'] == 'Red'] = 0
y2 = le.fit_transform(df2018['Label'].values)
X2 = StandardScaler().fit_transform(X2)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.40, random_state=369)


# Define the Custom_knn   
class Custom_knn:
    def __init__(self, number_neighbots_k, distance_parameter_p):  
        self.number_neighbots_k = number_neighbots_k
        self.distance_parameter_p = distance_parameter_p   
        self.X = None
        self.Labels = None
        
#    def distance(self, a,b):
#        return np.linalg.norm(a-b, ord = self.distance_parameter_p)
    
    def __str__(self):  
        return self.number_neighbots_k, self.distance_parameter_p                
    
    def fit(self, X, Labels):
        self.X = X
        self.Labels = Labels      
        return self
    
    def predict(self, new_x):
        assert self.X is not None
        assert self.Labels is not None
        predict = []
        for b in range(len(new_x)):
            dist = []
            for a in self.X:
                distance = np.linalg.norm(new_x[b]-a, ord = self.distance_parameter_p)
                dist.append(distance)
            sort = np.argsort(dist)
            topK = [self.Labels[i] for i in sort[:self.number_neighbots_k]]
            votes = Counter(topK)
            y_pred = votes.most_common(1)[0][0]
            predict.append(y_pred)
        return predict
        
    def draw_decision_boundary(self, new_x):
        dist = []
        for a in self.X:
            distance = np.linalg.norm(new_x-a, ord = self.distance_parameter_p)
            dist.append(distance)
        sort = np.argsort(dist)
            
        for i in sort[:self.number_neighbots_k]:
            print('id:',i, 'Label:',self.Labels[i])
         

# The optimal k value in the last assignment is k=3
# Question 1: Accuracy for different p
p = [1,1.5,2]
accuracy = []
for i in p:
    knn = Custom_knn(3,i)
    knn.fit(X_train, y_train)
    pred_p = knn.predict(X_test)
    accuracy.append(metrics.accuracy_score(y_test, pred_p))

# Plot accuracy 
plt.plot(p, accuracy, linestyle='dashed', marker='o')
plt.title('Accuracy for Different Distance')
plt.xlabel('p')
plt.xticks([1.0,1.5,2])
plt.ylabel('Accuracy')

# Find the distance metric gives the highest accuracy
accuracy_df = pd.DataFrame(accuracy, p)

optimal = 0
if accuracy_df.idxmax()[0] == 1:
    optimal = 'Manhattan'
elif accuracy_df.idxmax()[0] == 1.5:
    optimal = 'Minkovski'
else:
    optimal = 'Euclidean'
            
print('The distance metric that gives the highest accuracy is',optimal)



# Question 2
# accuracy for different p
p = [1,1.5,2]
accuracy2 = []
for i in p:
    knn = Custom_knn(3,i)
    knn.fit(X2_train, y2_train)
    pred_p2 = knn.predict(X2_test)
    accuracy2.append(metrics.accuracy_score(y2_test, pred_p2))
    
# Plot accuracy 
plt.plot(p, accuracy2, linestyle='dashed', marker='o')
plt.title('Accuracy for Different Distance')
plt.xticks([1.0,1.5,2])
plt.xlabel('p')
plt.ylabel('Accuracy')

# Find the distance metric gives the highest accuracy
accuracy_df2 = pd.DataFrame(accuracy2, p)

optimal2 = 0
if accuracy_df2.idxmax()[0] == 1:
    optimal2 = 'Manhattan'
elif accuracy_df2.idxmax()[0] == 1.5:
    optimal2 = 'Minkovski'
else:
    optimal2 = 'Euclidean'
            
print('The distance metric that gives the highest accuracy is',optimal2)


# Question 3
# p=1.5
knn2 = Custom_knn(3, 1.5)
knn2.fit(X_train, y_train)
y_pred = knn2.predict(X2)
# Index = 1: label = 1 (red)
print('For Week 4, the neighbors are:')
knn2.draw_decision_boundary(X2[3])
# Index = 5: label = 0 (green)
print('For Week 6, the neighbors are:')
knn2.draw_decision_boundary(X2[5])


# Question 4
def confusion_matrix_p(p):
    knn = Custom_knn(3,p)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X2)
    return confusion_matrix(y2, y_pred)
    
print('For p=1 the confusion matrix is: \n',confusion_matrix_p(1) )  
print('For p=1.5 the confusion matrix is: \n',confusion_matrix_p(1.5))  
print('For p=2 the confusion matrix is: \n',confusion_matrix_p(2))
  

# Question 5: TPR & TNR
def report(p):
    knn = Custom_knn(3,p)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X2)
    return classification_report(y2, y_pred)

def TPR(p):
    knn = Custom_knn(3,p)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X2)
    recall = recall_score(y2, y_pred)
    return print('For p=',p,'The true positive rate (sensitivity or recall) for 2018 is',round(recall,2))

def TNR(p):
    cm = confusion_matrix_p(p)
    TNR = cm[0,0]/(cm[0,0] + cm[0,1]) 
    print('For p=',p,'The true negative rate (speciﬁcity) for 2018 is',round(TNR,2))
    

report(1)
TPR(1)
TNR(1)
report(1.5)
TPR(1.5)
TNR(1.5)
report(2)
TPR(2)
TNR(2)
print('There are differences for different distance methods.')


# Question 6
# Buy-and-hold strategy
share = 100/df2018['Adj Close'][0]
value = share * df2018['Adj Close'][51]
print('For Buy-and-hold strategy, the final amount is $',round(value,2))

# New strategy
def new_strategy(p):
    knn = Custom_knn(3,p)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X2)
    
    position = 0
    share = 100/df2018['Adj Close'][0]
    cash = 0
    for i in range(len(y_pred)-1):
        if y_pred[i+1] == 0:
            # 0 refers to red
            if position != 0:
                cash += share * df2018['Adj Close'][i]
                share = 0
                position = 0
        elif y_pred[i+1] == 1:
            # 1 refers to green
            if position == 0:
                share += cash/df2018['Adj Close'][i]
                cash = 0
                position = 1
    value_new = cash + share * df2018['Adj Close'][51]
    return value_new

print('For the new trading strategy:')
print('When p=1, the final amount is $',round(new_strategy(1),2))
print('When p=1.5, the final amount is $',round(new_strategy(1.5),2))
print('When p=2, the final amount is $',round(new_strategy(2),2))

# Comparison
compare = [new_strategy(1),new_strategy(1.5),new_strategy(2),value]
if new_strategy(1) == max(compare):
    print('p=1 results in the largest portfolio value at the end of the year.')
elif new_strategy(1.5) == max(compare):
    print('p=1.5 results in the largest portfolio value at the end of the year.')
elif new_strategy(2) == max(compare):
    print('p=2 results in the largest portfolio value at the end of the year.')
else:
    print('The buy-and-hold strategy results in the largest portfolio value at the end of the year.')





 
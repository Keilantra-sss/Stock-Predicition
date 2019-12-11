"""
Created on Sun Nov  3 15:43:33 2019

@author: keilantra
"""


import os
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
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



# Compute the contribution of μ and σ for logistic regression

# Test accuracy using mean and std
# Preprocessing data
feature = ['mean', 'std']
X = df2017[feature].values
df2017['Label'][df2017['Label'] == 'Green'] = 1
df2017['Label'][df2017['Label'] == 'Red'] = 0
y = df2017.loc[:, 'Label'].values
le = LabelEncoder()
X_train = StandardScaler().fit_transform(X)
y_train = le.fit_transform(y)
# Logistic regression
log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(X_train,y_train)
# Calculate accuracy
X2 = df2018[feature].values
y2 = le.fit_transform(df2018['Label'].values)
y_pred = log_reg_classifier.predict(X2)
accuracy1 = np.mean(y_pred == y2)
print('The accuracy (using mean and std) is',round(accuracy1,2))

# Test accuracy using mean only
# Preprocessing data
feature = ['mean']
X = df2017[feature].values
df2017['Label'][df2017['Label'] == 'Green'] = 1
df2017['Label'][df2017['Label'] == 'Red'] = 0
y = df2017.loc[:, 'Label'].values 
le = LabelEncoder()
X_train = StandardScaler().fit_transform(X)
y_train = le.fit_transform(y)
# Logistic regression
log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(X_train,y_train)
# Calculate accuracy
X2 = df2018[feature].values
y2 = le.fit_transform(df2018['Label'].values)
y_pred = log_reg_classifier.predict(X2)
accuracy2 = np.mean(y_pred == y2)
print('The accuracy (using mean only) is',round(accuracy2,3))


# Test accuracy using std only
# Preprocessing data
feature = ['std']
X = df2017[feature].values
df2017['Label'][df2017['Label'] == 'Green'] = 1
df2017['Label'][df2017['Label'] == 'Red'] = 0
y = df2017.loc[:, 'Label'].values
le = LabelEncoder()
X_train = StandardScaler().fit_transform(X)
y_train = le.fit_transform(y)
# Logistic regression
log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(X_train,y_train)
# Calculate accuracy
X2 = df2018[feature].values
y2 = le.fit_transform(df2018['Label'].values)
y_pred = log_reg_classifier.predict(X2)
accuracy3 = np.mean(y_pred == y2)
print('The accuracy (using std only) is',round(accuracy3,3))



# Compute the contribution of μ and σ for Euclidean kNN
# The optimal k value of previous assignments is 3

# Test accuracy using mean and std
# Preprocessing data
feature = ['mean', 'std']
X = df2017[feature].values
df2017['Label'][df2017['Label'] == 'Green'] = 1
df2017['Label'][df2017['Label'] == 'Red'] = 0
y = df2017.loc[:, 'Label'].values
le = LabelEncoder()
X_train = StandardScaler().fit_transform(X)
y_train = le.fit_transform(y)
# Update kNN model
knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(X_train, y_train)
# Calculate accuracy
X2 = df2018[feature].values
le = LabelEncoder()
df2018['Label'][df2018['Label'] == 'Green'] = 1
df2018['Label'][df2018['Label'] == 'Red'] = 0
y2 = le.fit_transform(df2018['Label'].values)
X2 = StandardScaler().fit_transform(X2)
y_pred = knn.predict(X2)
accuracy4 = metrics.accuracy_score(y2, y_pred)
print('The accuracy (using mean and std) is', round(accuracy4, 2))


# Test accuracy using mean only
# Preprocessing data
feature = ['mean']
X = df2017[feature].values
df2017['Label'][df2017['Label'] == 'Green'] = 1
df2017['Label'][df2017['Label'] == 'Red'] = 0
y = df2017.loc[:, 'Label'].values 
le = LabelEncoder()
X_train = StandardScaler().fit_transform(X)
y_train = le.fit_transform(y)
# Update kNN model
knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(X_train, y_train)
# Calculate accuracy
X2 = df2018[feature].values
le = LabelEncoder()
df2018['Label'][df2018['Label'] == 'Green'] = 1
df2018['Label'][df2018['Label'] == 'Red'] = 0
y2 = le.fit_transform(df2018['Label'].values)
X2 = StandardScaler().fit_transform(X2)
y_pred = knn.predict(X2)
accuracy5 = metrics.accuracy_score(y2, y_pred)
print('The accuracy (using mean only) is', round(accuracy5, 3))


# Test accuracy using std only
# Preprocessing data
feature = ['std']
X = df2017[feature].values
df2017['Label'][df2017['Label'] == 'Green'] = 1
df2017['Label'][df2017['Label'] == 'Red'] = 0
y = df2017.loc[:, 'Label'].values
le = LabelEncoder()
X_train = StandardScaler().fit_transform(X)
y_train = le.fit_transform(y)
# Update kNN model
knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(X_train, y_train)
# Calculate accuracy
X2 = df2018[feature].values
le = LabelEncoder()
df2018['Label'][df2018['Label'] == 'Green'] = 1
df2018['Label'][df2018['Label'] == 'Red'] = 0
y2 = le.fit_transform(df2018['Label'].values)
X2 = StandardScaler().fit_transform(X2)
y_pred = knn.predict(X2)
accuracy6 = metrics.accuracy_score(y2, y_pred)
print('The accuracy (using std only) is', round(accuracy6, 3))


# Summarize the result in the table
log_reg = [accuracy1, accuracy2, accuracy3]
kNN = [accuracy4, accuracy5, accuracy6]
c = {'log_reg':log_reg, 'kNN':kNN}
result = pd.DataFrame(c, index=['mean and std','mean only','std only'])
print(result)

# Discuss the findings
print('For logistic regression, using only mean can contribute to the highest accuracy, while using both mean and std would result in the lowest accuracy. Thus for this model, we do not need them both.')
print('For kNN model, using both mean and std can contribute to the highest accuracy, while using only std would result in the lowest accuracy. Thus for this model, we need them both to increase the accuracy.')




 



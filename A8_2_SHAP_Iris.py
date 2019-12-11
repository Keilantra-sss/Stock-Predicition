#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 12:45:42 2019

@author: keilantra
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def SHAP(iris, varname):
#for iris in ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'):
#    for varname in ('sepal-length', 'sepal-width', 'petal-length', 'petal-width', ''):
    
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    data = pd.read_csv(url, names=['sepal-length', 'sepal-width',
                                   'petal-length', 'petal-width', 'Class'])
    feature = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width']
    try:
        feature.remove(varname)
    except:
        pass
    X = data[feature].values
    data['Class'][data['Class'] == iris] = 1
    data['Class'][data['Class'] != 1] = 0
    y = data.loc[:,'Class'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=3)

    log_reg_classifier = LogisticRegression()
    log_reg_classifier.fit(X_train, y_train.astype('int'))

    y_pred = log_reg_classifier.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    return accuracy
          
#print('The accuracy for '+iris+' without '+varname+' is ', round(accuracy, 2))

# For Versicolor
X1 = SHAP('Iris-versicolor','')
Y1_sl = SHAP('Iris-versicolor','sepal-length')
Y1_sw = SHAP('Iris-versicolor','sepal-width')
Y1_pl = SHAP('Iris-versicolor','petal-length')
Y1_pw = SHAP('Iris-versicolor','petal-width')
    
sl1 = X1 - Y1_sl  
sw1 = X1 - Y1_sw
pl1 = X1 - Y1_pl
pw1 = X1 - Y1_pw

Versicolor = [sl1,sw1,pl1,pw1]

# For Setosa
X2 = SHAP('Iris-setosa','')
Y2_sl = SHAP('Iris-setosa','sepal-length')
Y2_sw = SHAP('Iris-setosa','sepal-width')
Y2_pl = SHAP('Iris-setosa','petal-length')
Y2_pw = SHAP('Iris-setosa','petal-width')
    
sl2 = X2 - Y2_sl  
sw2 = X2 - Y2_sw
pl2 = X2 - Y2_pl
pw2 = X2 - Y2_pw

Setosa = [sl2,sw2,pl2,pw2]

# For Virginica
X3 = SHAP('Iris-virginica','')
Y3_sl = SHAP('Iris-virginica','sepal-length')
Y3_sw = SHAP('Iris-virginica','sepal-width')
Y3_pl = SHAP('Iris-virginica','petal-length')
Y3_pw = SHAP('Iris-virginica','petal-width')
    
sl3 = X3 - Y3_sl  
sw3 = X3 - Y3_sw
pl3 = X3 - Y3_pl
pw3 = X3 - Y3_pw

Virginica = [sl3,sw3,pl3,pw3]


# Summarize the result in the table
c = {'Versicolor':Versicolor,'Setosa':Setosa,'Virginica':Virginica}
result = pd.DataFrame(c, index=['sepal length ∆','sepal width ∆','petal length ∆','petal width ∆'])
print(result)












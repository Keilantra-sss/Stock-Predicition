#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import copy
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score

# Import data
try:
    df = pd.read_csv(r'/Users/keilantra/Desktop/bank.csv')  # change the path
except Exception as e:
    print(e)
    print('Failed to read the csv data')

# ======================== Data Exploration ========================
df.head()
df.describe()
df.dtypes
print('The dataset has numerical and categorical data.')

# Remove some rows with 'unknown'
missing_val_cols = ['education','job']
for col in missing_val_cols:    
    df = df[df[col] != 'unknown']

# Clean the data
df.isnull().sum()
print('There is no null values, the data is clean')

# From the instruction, 'duration' is best to be omitted
'''
duration: the last contact duration

Although this attribute highly affects the output target,
the duration is not known before a call is performed, 
also, after the end of the call y is obviously known.
Thus, this input should be discarded when having a realistic predictive model.
'''
df = df.drop('duration', axis=1)

# Numerical data exploration
num_cols  = df.columns.values[(df.dtypes =='int64')]
num_cols

fig, axs = plt.subplots(2, 3, sharex=False, sharey=False, figsize=(20, 15))
counter = 0
for cols in num_cols:   
    trace_x = counter // 3
    trace_y = counter % 3    
    axs[trace_x, trace_y].hist(df[cols])    
    axs[trace_x, trace_y].set_title(cols)    
    counter += 1
plt.show()

# Check the noise data
df[['pdays', 'campaign', 'previous']].describe()
len(df[df['pdays']>400])/len(df)*100     # drop this variable
len(df[df['campaign']>20])/len(df)*100   # replace the outliers
len(df[df['previous']>15])/len(df)*100   # replace the outliers

# Check for numerical features
num_cols = np.delete(num_cols,4)
corrmat = df[num_cols].corr(method='spearman')
sns.heatmap(corrmat, annot=True,cbar=False, square=True)


# Categorical data exploration
# Analysis of the outcome variable
value_counts = df['deposit'].value_counts()
value_counts.plot.bar(title='Deposit value counts')

# Analysis: job - deposit
j_df = pd.DataFrame()
j_df['yes'] = df[df['deposit'] == 'yes']['job'].value_counts()
j_df['no'] = df[df['deposit'] == 'no']['job'].value_counts()
j_df.plot.bar(title = 'Job - Deposit')
# Analysis: marital - deposit
j_df = pd.DataFrame()
j_df['yes'] = df[df['deposit'] == 'yes']['marital'].value_counts()
j_df['no'] = df[df['deposit'] == 'no']['marital'].value_counts()
j_df.plot.bar(title = 'Marital - Deposit')
# Analysis: education - deposit
j_df = pd.DataFrame()
j_df['yes'] = df[df['deposit'] == 'yes']['education'].value_counts()
j_df['no'] = df[df['deposit'] == 'no']['education'].value_counts()
j_df.plot.bar(title = 'Education - Deposit')
# Analysis: type of the contact - deposit
j_df = pd.DataFrame()
j_df['yes'] = df[df['deposit'] == 'yes']['contact'].value_counts()
j_df['no'] = df[df['deposit'] == 'no']['contact'].value_counts()
j_df.plot.bar(title = 'Type of Contact - Deposit')
# Analysis: month - deposit
j_df = pd.DataFrame()
j_df['yes'] = df[df['deposit'] == 'yes']['month'].value_counts()
j_df['no'] = df[df['deposit'] == 'no']['month'].value_counts()
j_df.plot.bar(title = 'Month - Deposit')
# ======================== Data Exploration End=====================



# ======================== Data Preparation ========================
df1 = copy.copy(df)
df1 = df1.drop('deposit', axis=1)
df1 = df1.drop('pdays', axis=1)

# Replace the outliers
def get_correct_values(row, column_name, threshold, df):
    if row[column_name] <= threshold:
        return row[column_name]
    else:
        mean = df[df[column_name] <= threshold][column_name].mean()
        return mean
df1['campaign*'] = df1.apply(lambda row: get_correct_values(row, 'campaign', 20, df1),axis=1)
df1['previous*'] = df1.apply(lambda row: get_correct_values(row, 'previous', 15, df1),axis=1)
df1 = df1.drop(['campaign', 'previous'], axis=1)
 
# Convert age to categorical variable           
df1['age_bin'] = pd.cut(df['age'], [0, 20, 30, 40, 50, 60, 70, 80, 90, 100], 
                       labels=['0-20', '20-30', '30-40', '40-50','50-60','60-70','70-80', '80-90','90-100'])
df1 = df1.drop('age',axis = 1)

# Convert categorical columns to binary variables (0 - no, 1 - yes)
le = LabelEncoder()
df1['default'] = le.fit_transform(df1['default']) 
df1['housing'] = le.fit_transform(df1['housing']) 
df1['loan'] = le.fit_transform(df1['loan']) 

# Convert categorical columns into dummy variables
cat_cols = ['job', 'marital', 'education', 'contact', 'poutcome', 'age_bin','month'] 
for col in cat_cols:
    dummies = pd.get_dummies(df1[col], prefix=col, prefix_sep='_', 
                             drop_first=True, dummy_na=False)
    df1 = pd.concat([df1.drop(col, axis=1), dummies], axis=1)

df1.head()

# Split the data
X = df1
X = StandardScaler().fit_transform(X)
y = le.fit_transform(df['deposit'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=3)
# ======================== Data Preparation End =======================



# ======================== XGBoost Classifier =========================
# XGBoost model
xgb = XGBClassifier(n_estimators=200, learning_rate=0.08, gamma=0.3, subsample=0.9,
                    colsample_bytree=0.7, max_depth=6, min_child_weight=5)
xgb.fit(X_train,y_train)

y_train_preds = xgb.predict(X_train)
y_test_preds = xgb.predict(X_test)

print('Accuracy score: train: %.3f: test: %.3f' % (
metrics.accuracy_score(y_train, y_train_preds),
        metrics.accuracy_score(y_test, y_test_preds)))

# Get feature importances f
headers = ['name', 'score']
values = sorted(zip(df1.columns, xgb.feature_importances_), key=lambda x: x[1] * -1)
xgb_feature_importances = pd.DataFrame(values, columns = headers)

# Plot feature importances
x_pos = np.arange(0, len(xgb_feature_importances))
plt.bar(x_pos, xgb_feature_importances['score'])
plt.xticks(x_pos, xgb_feature_importances['name'])
plt.xticks(rotation=90)
plt.title('Feature importances (XGB)')
plt.show()
# ======================== XGBoost Classifier End =======================



# ======================== Update Data Preparation ======================
# Update the features
features = ['poutcome','contact','housing','month','age']

df2 = copy.copy(df[features])

# Convert age to categorical variable           
df2['age_bin'] = pd.cut(df['age'], [0, 20, 30, 40, 50, 60, 70, 80, 90, 100], 
                       labels=['0-20', '20-30', '30-40', '40-50','50-60','60-70','70-80', '80-90','90-100'])
df2 = df2.drop('age',axis = 1)

# Convert categorical columns to binary variables (0 - no, 1 - yes)
df2['housing'] = le.fit_transform(df2['housing']) 

# Convert categorical columns into dummy variables
cat_cols = ['contact', 'poutcome', 'age_bin','month'] 
for col in cat_cols:
    dummies = pd.get_dummies(df2[col], prefix=col, prefix_sep='_', 
                             drop_first=True, dummy_na=False)
    df2 = pd.concat([df2.drop(col, axis=1), dummies], axis=1)

# Preprocessing
X = df2.values
X = StandardScaler().fit_transform(X)
y = le.fit_transform(df['deposit'])

# Split the data into train, valid and test set
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.40, random_state=3)
X_valid,X_test,y_valid,y_test = train_test_split(X_valid, y_valid, test_size=0.5,random_state=3)
# ======================== Update Data Preparation End ===================



# ======================== kNN Algorithm ==============================
# Find the best k
accuracy = []
for n in range(1,100,2):
    knn = KNeighborsClassifier(n_neighbors=n, metric='euclidean')
    knn.fit(X_train, y_train)
    pred_k = knn.predict(X_valid)
    accuracy.append(metrics.accuracy_score(y_valid, pred_k))

# Visuliaze the accuracy of different k values
plt.plot(np.arange(1,100,2), accuracy, linestyle='dashed')
plt.title('Accuracy for K Value')
plt.xlabel('K Value')
plt.ylabel('Accuracy')

accuracy_df = pd.DataFrame(accuracy, np.arange(1,100,2))
optimal = accuracy_df.idxmax()
print('The optimal value of k is',optimal[0]) # k=17

# Perform the knn using optimal values
knn = KNeighborsClassifier(n_neighbors=optimal[0], metric='euclidean')
knn.fit(X_train, y_train)
y_pred1 = knn.predict(X_test)

# Evaluate the classifier
accuracy1 = metrics.accuracy_score(y_test, y_pred1)
precision1 = precision_score(y_test, y_pred1, average='weighted')
recall1 = recall_score(y_test, y_pred1, average='weighted')
f11 = metrics.f1_score(y_test,y_pred1, average='weighted')

# Summarize the result in the table
result1 = [accuracy1, precision1, recall1, f11]
a1 = {'kNN':result1}
performance_knn = pd.DataFrame(a1, index=['accuracy','precision','recall','f1-score'])
print(round(performance_knn,5))
# ======================== kNN Algorithm End ============================



# ======================== Random Forest ================================
# Find the best combination of N and d 
result = pd.DataFrame(index = range(1,50),columns = range(1,50))
for N in range(1,50):
    for d in range(1,50):
        clf = RandomForestClassifier(n_estimators=N, max_depth=d,\
                                            criterion="entropy", random_state=0)
        clf.fit(X_train, y_train)
        pred_nd= clf.predict(X_valid)
        accuracy = metrics.accuracy_score(y_valid, pred_nd)
        result.iloc[N-1,d-1] = accuracy
        
best = sorted(result.max())[0]
for indexs in result.index:
    for i in range(len(result.loc[indexs].values)):
        if result.loc[indexs].values[i] == best:
            N = indexs
            d = i+1
print('The best combination is N=',N,'d=',d)

# Perform the random forest classifier using optimal values
clf = RandomForestClassifier(n_estimators=N, max_depth=d, criterion='entropy')
# N=49, d=1
clf.fit(X_train,y_train)
y_pred2 = clf.predict(X_test) 

# Evaluate the classifier
accuracy2 = metrics.accuracy_score(y_test, y_pred2)
precision2 = precision_score(y_test, y_pred2, average='weighted')
recall2 = recall_score(y_test, y_pred2, average='weighted')
f12 = metrics.f1_score(y_test, y_pred2, average='weighted')

# Summarize the result in the table
result2 = [accuracy2, precision2, recall2, f12]
a2 = {'Random Forest': result2}
performance_rf = pd.DataFrame(a2, index=['accuracy','precision','recall','f1-score'])
print(round(performance_rf,5))
# ======================== Random Forest End ============================



# ======================== Support Vector Machine =======================
# Perform the linear SVM
svm_l = svm.SVC(kernel='linear')
svm_l.fit(X_train, y_train)
y_pred3_l = svm_l.predict(X_valid)
accuracy3_l = metrics.accuracy_score(y_valid, y_pred3_l)
print('The accuracy for linear SVM is',round(accuracy3_l,3))

# Perform the Gaussian SVM
svm_g = svm.SVC(kernel='rbf')
svm_g.fit(X_train, y_train)
y_pred3_g = svm_g.predict(X_valid)
accuracy3_g = metrics.accuracy_score(y_valid, y_pred3_g)
print('The accuracy for Gaussian SVM is',round(accuracy3_g,3))

# Perform the polynomial SVM
svm_p = svm.SVC(kernel='poly',degree = 2)
svm_p.fit(X_train, y_train)
y_pred3_p = svm_p.predict(X_valid)
accuracy3_p = metrics.accuracy_score(y_valid, y_pred3_p)
print('The accuracy for polynomial SVM is',round(accuracy3_p,3))

# Compare the accuracy for SVM
compare = [accuracy3_l,accuracy3_g,accuracy3_p]
if accuracy3_l == max(compare):
    print('The linear method is better.')
    method = 'linear'
elif accuracy3_g == max(compare):
    print('The Gaussian method is better.')
    method = 'rbf'
else:
    print('The polynomial method is better.')
    method = 'poly'

# Perform the SVM classifier using the best method
svm = svm.SVC(kernel=method) # Gaussian
svm.fit(X_train, y_train)
y_pred3 = svm_p.predict(X_test)
accuracy3 = metrics.accuracy_score(y_test, y_pred3)
precision3 = metrics.precision_score(y_test, y_pred3, average='weighted')
recall3 = metrics.recall_score(y_test, y_pred3, average='weighted')
f13 = metrics.f1_score(y_test, y_pred3, average='weighted')

# Summarize the result in the table
result3 = [accuracy3, precision3, recall3, f13]
a3 = {'SVM': result3}
performance_svm = pd.DataFrame(a3, index=['accuracy','precision','recall','f1-score'])
print(round(performance_svm,5))
# ======================== Support Vector Machine End ====================


# Summarize the result in the table
performance = pd.concat([performance_knn, performance_rf, performance_svm], axis=1)
print('The performance for different algorithms is: \n')
print(round(performance,3))

# Compare the accuracy for different algorithms
compare2 = [accuracy1,accuracy2,accuracy3]
if accuracy1 == max(compare2):
    print('The kNN algorithm is the best choice.')
elif accuracy2 == max(compare2):
    print('The Random Forest algorithm is the best choice.')
else:
    print('The Gaussian SVM method is the best choice.')
# the best choice is kNN


# Make a example prediciton using the best     
client1 = [1,1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
'''
The client has housing loan. 
The client uses telephone to contact.
The previous marketing campaign for this client is success.
The client is 24 years old.
The month is May.
'''
client2 = [0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
'''
The client has no housing loan. 
The client uses cellular to contact.
The previous marketing campaign for this client is unknown.
The client is 34 years old.
The month is May.
'''
client3 = [1,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
'''
The client has housing loan. 
The client uses telephone to contact.
The previous marketing campaign for this client is failure.
The client is 44 years old.
The month is Spetember.
'''

X2 = pd.DataFrame([client1, client2, client3]).values
X2 = StandardScaler().fit_transform(X2)
y_pred = knn.predict(X2)

result3 = [accuracy3, precision3, recall3, f13]
a4 = {'deposit': y_pred}
example = pd.DataFrame(a4, index=['client1','client2','client3'])
print(example)


    

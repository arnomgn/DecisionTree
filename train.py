import pandas as pd
import pydot
from sklearn import tree
from time import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from operator import itemgetter

#Data Preprocessing
train_set = pd.read_csv('data/train.csv')
test_set = pd.read_csv('data/test.csv')
train_df = train_set.iloc[train_set.index]


targets = train_set['Category'].unique()
print(len(targets))
'''
map_to_int = {name: n for n , name in enumerate(targets)}

days = train_set['DayOfWeek'].unique()
days_int = {label: n for n, label in enumerate(days)}
district = train_set['PdDistrict'].unique()
district_int = {label: n for n, label in enumerate(district)}

train_df['Dates'] = pd.to_datetime(train_df['Dates'])
train_df['Year'] = train_df['Dates'].dt.year
train_df['Month'] = train_df['Dates'].dt.month
train_df['Hour'] = train_df['Dates'].dt.hour

train_df_copy = train_df.copy()
train_df_copy['Category'] =  train_df_copy['Category'].replace(map_to_int)
train_df_copy['DayOfWeek'] = train_df_copy['DayOfWeek'].replace(days_int)
train_df_copy['PdDistrict'] = train_df_copy['PdDistrict'].replace(district_int)

#Training

features = list( train_df_copy.columns[i] for i in [3,4,7,8, 9, 10, 11])

y=train_df_copy['Category']
x=train_df_copy[features]

dtree =DecisionTreeClassifier(min_samples_split=10000,random_state=99)
dtree.fit(x,y)


# Test Data Preprocessing
sample = pd.read_csv('data/sample_submission.csv')
sample_df = sample

test_df = test_set.iloc[test_set.index]
test_df['DayOfWeek'] = test_df['DayOfWeek'].replace(days_int)
test_df['PdDistrict'] = test_df['PdDistrict'].replace(district_int)
test_df['Dates'] = pd.to_datetime(test_df['Dates'])
test_df['Year'] = test_df['Dates'].dt.year
test_df['Month'] = test_df['Dates'].dt.month
test_df['Hour'] = test_df['Dates'].dt.hour



#Test

x_test = test_df[features]
predictedValues = dtree.predict_proba(x_test)

#Produce the output

output = pd.DataFrame(data = predictedValues, columns = targets)
output.index.names = ['Id']

output.reset_index()
output.to_csv('output.csv', index=True)
'''
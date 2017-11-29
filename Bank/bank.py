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

targets = train_set['y']

jobs = train_set['job'].unique()
jobs_int = {label: n for n, label in enumerate(jobs)}

marital = train_set['marital'].unique()
marital_int = {label: n for n, label in enumerate(marital)}

education = train_set['education'].unique()
education_int = {label: n for n, label in enumerate(education)}

default = train_set['default'].unique()
default_int = {label: n for n, label in enumerate(default)}

housing = train_set['housing'].unique()
housing_int = {label: n for n, label in enumerate(housing)}

loan = train_set['loan'].unique()
loan_int = {label: n for n, label in enumerate(loan)}

contact = train_set['contact'].unique()
contact_int = {label: n for n, label in enumerate(contact)}

month = train_set['month'].unique()
month_int = {label: n for n, label in enumerate(month)}

poutcome = train_set['poutcome'].unique()
poutcome_int = {label: n for n, label in enumerate(poutcome)}

train_df_copy = train_df.copy()
train_df_copy['job'] =  train_df_copy['job'].replace(jobs_int)
train_df_copy['marital'] =  train_df_copy['marital'].replace(marital_int)
train_df_copy['education'] =  train_df_copy['education'].replace(education_int)
train_df_copy['default'] =  train_df_copy['default'].replace(default_int)
train_df_copy['housing'] =  train_df_copy['housing'].replace(housing_int)
train_df_copy['loan'] =  train_df_copy['loan'].replace(loan_int)
train_df_copy['contact'] =  train_df_copy['contact'].replace(contact_int)
train_df_copy['month'] =  train_df_copy['month'].replace(month_int)
train_df_copy['poutcome'] =  train_df_copy['poutcome'].replace(poutcome_int)


#Training
features = list( train_df_copy.columns[i] for i in range(0,16))

y=train_df_copy['y']
x=train_df_copy[features]

dtree =DecisionTreeClassifier(min_samples_split=2,criterion='entropy')
dtree.fit(x,y)

#Test Preprocessing
test_df = test_set.iloc[test_set.index]

test_df['job'] =  test_df['job'].replace(jobs_int)
test_df['marital'] =  test_df['marital'].replace(marital_int)
test_df['education'] =  test_df['education'].replace(education_int)
test_df['default'] =  test_df['default'].replace(default_int)
test_df['housing'] =  test_df['housing'].replace(housing_int)
test_df['loan'] =  test_df['loan'].replace(loan_int)
test_df['contact'] =  test_df['contact'].replace(contact_int)
test_df['month'] =  test_df['month'].replace(month_int)
test_df['poutcome'] =  test_df['poutcome'].replace(poutcome_int)

#Test
x_test = test_df[features]
predictedValues = dtree.predict(x_test)


#Produce Output
output = pd.DataFrame(data = predictedValues, columns = ['y'])
output.index = output.index +1
output.index.names = ['id']


output.reset_index()
output.to_csv('output.csv', index=True)

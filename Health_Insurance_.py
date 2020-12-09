# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 13:38:29 2020

@author: shiva dumnawar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv('Health_insurance.csv')

df.info()

df.isnull().sum() # no null values

df.describe()

# check outliers
df.plot(kind= 'box')

# remove outliers

df['bmi']= df['bmi'].clip(lower= df['bmi'].quantile(0.1), upper= df['bmi'].quantile(0.9) )

df['charges']= df['charges'].clip(lower= df['charges'].quantile(0.15), upper= df['charges'].quantile(0.85) )

df.plot(kind= 'box')

# one hot encoding
df= pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first= True)

df.columns
new_order= ['age', 'bmi', 'children', 'sex_male', 'smoker_yes',
       'region_northwest', 'region_southeast', 'region_southwest', 'charges']

df= df[new_order]

# correlation
plt.figure(figsize=(10,8))
c= df.corr()
sns.heatmap(c, cmap='coolwarm', annot=True)
plt.xticks(rotation=30)
plt.tight_layout()

X= df.iloc[:, :-1]
y= df.iloc[:, -1].values.reshape(-1,1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=14)

# standardization
from sklearn.preprocessing import StandardScaler
ss= StandardScaler()

cols=['age', 'bmi', 'children']

X_train[cols]= ss.fit_transform(X_train[cols])

X_test[cols]= ss.fit_transform(X_test[cols])

from sklearn.linear_model import LinearRegression
model= LinearRegression()
model.fit(X_train, y_train)

pred= model.predict(X_test)

model.score(X_train, y_train)
model.score(X_test, y_test)

from sklearn.metrics import r2_score, mean_squared_error

r2_score(y_test, pred)

mean_squared_error(y_test, pred)
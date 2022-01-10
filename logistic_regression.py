import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()

row_data = pd.read_csv('dataset.csv')

StandardScaler = StandardScaler()  

columns_to_scale = ['age','trestbps','chol','thalach','oldpeak']
row_data[columns_to_scale] = StandardScaler.fit_transform(row_data[columns_to_scale])

X= row_data.drop(['target'], axis=1)
y= row_data['target']

X_train, X_test,y_train, y_test=train_test_split(X,y,test_size=0.3)

print('X_train-', X_train.size)
print('X_test-',X_test.size)
print('y_train-', y_train.size)
print('y_test-', y_test.size)   

#poia h shmasia tou posostoy twn train kai test kai ti diafores uparxoun kai pos vriskoume to idaniko
model1=lr.fit(X_train,y_train)
prediction1=model1.predict(X_test)

print(accuracy_score(y_test,prediction1))



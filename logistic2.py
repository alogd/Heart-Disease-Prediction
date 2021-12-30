import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import data_manipulation as dm
from  itertools  import combinations
import numpy as np
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

lr=LogisticRegression()

data=dm.get_dataframe(['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'])
X= data.drop(['target'], axis=1)
y= data['target']


sfs1 = SFS(LogisticRegression(),
         k_features=(1,15),
         forward=False,
         floating=False,
         cv=0)

sfs1.fit(X, y)
print(sfs1.k_score_)
print(sfs1.k_feature_names_)
print("=========custom regression")
desired_columns=['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca', 'thal', 'cp_0', 'cp_3', 'exang_0']
for col in X.columns:
    if col not in desired_columns:
        X=X.drop([col],1)

print(X.head(5))
lr=LogisticRegression()
X_train, X_test,y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=40)


#poia h shmasia tou posostoy twn train kai test kai ti diafores uparxoun kai pos vriskoume to idaniko
model1=lr.fit(X_train,y_train)
prediction1=model1.predict(X_test)

print(accuracy_score(y_test,prediction1))

import pandas as pd
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import data_manipulation as dm

wanted_columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']



X,y=dm.get_dataframe(wanted_columns, "oversampling" )


X_train, X_test,y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=40)

print('X_train-', X_train.size)
print('X_test-',X_test.size)
print('y_train-', y_train.size)
print('y_test-', y_test.size)


clf = RandomForestClassifier()

clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))


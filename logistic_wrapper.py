import pandas as pd
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LogisticRegression
import data_manipulation as dm

wanted_columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
lr=LogisticRegression()


X,y=dm.get_dataframe(wanted_columns, 'oversampling')

print('\n\nLogistic Regression Results:')

#Step Forward Feature Selection / Wrapper
sfs1 = SFS(LogisticRegression(),
         k_features=(1,10),
         forward=True,
         floating=False,
         cv=0)

print("\nStep Forward Selection + Oversampling")
sfs1.fit(X, y)
print(sfs1.k_score_)
print(sfs1.k_feature_names_)


#Step Backward Feature Selection / Wrapper
sfs1 = SFS(LogisticRegression(),
         k_features=(1,10),
         forward=False,
         floating=False,
         cv=0)

print("\nStep Backward Selection + Oversampling")
sfs1.fit(X, y)
print(sfs1.k_score_)
print(sfs1.k_feature_names_)



#Exhaustive Feature Selection / Wrapper 
efs = EFS(LogisticRegression(),
           min_features=1,
           max_features=3,
           scoring='accuracy',
           cv=2)
print("\n\nExhaustive Selection + Oversampling")

efs.fit(X,y)
print('\n',efs.best_score_)
print(efs.best_feature_names_)



#############################          Undersampling 

X,y=dm.get_dataframe(wanted_columns, 'undersampling')

#Step Forward Feature Selection / Wrapper
sfs1 = SFS(LogisticRegression(),
         k_features=(1,10),
         forward=True,
         floating=False,
         cv=0)
print("\nStep Forward Selection + Undersampling")
sfs1.fit(X, y)
print(sfs1.k_score_)
print(sfs1.k_feature_names_)


#Step Backward Feature Selection / Wrapper
sfs1 = SFS(LogisticRegression(),
         k_features=(1,10),
         forward=False,
         floating=False,
         cv=0)
print("\nStep Backward Selection + Undersampling")
sfs1.fit(X, y)
print(sfs1.k_score_)
print(sfs1.k_feature_names_)



#Exhaustive Feature Selection / Wrapper 
efs = EFS(LogisticRegression(),
           min_features=1,
           max_features=3,
           scoring='accuracy',
           cv=2)
print("\n\nExhaustive Selection + Undersampling")

efs.fit(X,y)
print('\n',efs.best_score_)
print(efs.best_feature_names_)



'''
print("=========custom regression")
desired_columns=sfs1.k_feature_names_
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
'''
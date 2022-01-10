
from pandas.core.frame import DataFrame
from sklearn. feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import data_manipulation as dm

#NA VROYME POS VLEPOYME POIA FEATURES EXOYN EPILEGEI META TO SELECTION 
#NA GINEI MIA MORFOPOIHSH SAN TO LOGISTIC WRAPPER
#MIA MIKRH PSAKTIKI GIA TA EMBEDDED METHODS



'''
#--------------------------------------------CHI 2--------------------------------

wanted_columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
lr=LogisticRegression()

X,y=dm.get_dataframe(wanted_columns, 'oversampling', 0)


X_train, X_test,y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=40)

print(X.shape)

X_new = SelectKBest(chi2, k=7).fit_transform(X, y)
print(X_new.head)

'''



'''
 #---------------------VARIANCE THRESHOLD---------------------------

wanted_columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
lr=LogisticRegression()
sel = VarianceThreshold(threshold=0.5)
# na valoume ton tropo me ton opoio epilegei thn apolutow kauteri timi gia to threshold h test kathe score me
#diafores times threshold

X,y=dm.get_dataframe(wanted_columns, 'oversampling')

X_train, X_test,y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=40)

print(X_train.shape, X_test.shape)
print(X.shape)

sel.fit(X_train)

model1=lr.fit(X_train,y_train)
prediction1=model1.predict(X_test)
#to score tou monteloy mas prin ginei h afairesh ton features
print(accuracy_score(y_test,prediction1))

# h metatroph ton feature sta ligotera 
X_train = sel.transform(X_train)
X_test = sel.transform(X_test)

print(X_train.shape, X_test.shape)

model1=lr.fit(X_train,y_train)
prediction1=model1.predict(X_test)
#to score tou monteloy me ta ligotera features meta thn afairesh me VARIANCE THRESHOLD
print(accuracy_score(y_test,prediction1))
'''







'''
#---------------------------------ANOVA----------------------------


wanted_columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
lr=LogisticRegression()

X,y=dm.get_dataframe(wanted_columns, 'oversampling')


# Create an SelectKBest object to select features with two best ANOVA F-Values
fvalue_selector = SelectKBest(f_classif, k=6)

# Apply the SelectKBest object to the features and target
X_kbest = fvalue_selector.fit_transform(X, y)


# View results
print('Original number of features:', X.shape[1])
print('Reduced number of features:', X_kbest.shape[1])

'''





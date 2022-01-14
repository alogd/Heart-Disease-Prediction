
from scipy.sparse.construct import random
from sklearn. feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import f_classif
from statistics import mean
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import data_manipulation as dm



print("\n\n Embedded Methods Feature Selection \n")
lr=LogisticRegression()

X,y=dm.get_dataframe(sampling = "oversampling") 

X_train, X_test,y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=0)

selection = SelectFromModel(LogisticRegression(C=1, penalty='l2'))
selection.fit(X_train, y_train)

selected_features = X_train.columns[(selection.get_support())]

print(selected_features)













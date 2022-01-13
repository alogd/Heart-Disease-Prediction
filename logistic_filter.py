
from scipy.sparse.construct import random
from sklearn. feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import f_classif
from statistics import mean
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import data_manipulation as dm

# 1. NA VROYME POS VLEPOYME POIA FEATURES EXOYN EPILEGEI META TO SELECTION 
# 2. NA GINEI MIA MORFOPOIHSH SAN TO LOGISTIC WRAPPER
# 3. MIA MIKRH PSAKTIKI GIA TA EMBEDDED METHODS
# 4. NA FTIAKSOYME METHODOYS GIA NA TSEKAROUME TA APOTELESMATA WSTE NA BROYME TO IDANIKO K KAI THRESHOLD GIA (ANOVA, CHI 2) & (THRESHOLD)



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




 #---------------------VARIANCE THRESHOLD---------------------------
print("\n\nVariance Feature Selection")
lr=LogisticRegression()


for sampling in ['oversampling', 'undersampling']:
    X,y=dm.get_dataframe(sampling=sampling)
    #We try all the values for threshold between 0 and 1 with a 0.01 step 
    currentThreshold=0
    mean_value_per_threshold=[]

    while currentThreshold<=1:
        sel = VarianceThreshold(threshold=currentThreshold)
        currentThreshold+=0.01
        result_of_split=[]
        #We do 100 different splits and save the mean accuracy value
        for i in range(100):
            X_train, X_test,y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=i)
            sel.fit(X_train)

            #We cut out the under threshold variables
            X_train = sel.transform(X_train)
            X_test = sel.transform(X_test)
            #We train the model 
            model1=lr.fit(X_train,y_train)
            prediction1=model1.predict(X_test)
            #We save the score 
            result_of_split.append(accuracy_score(y_test,prediction1))
        #Save result for this threshold value
        mean_value_per_threshold.append(mean(result_of_split))

    #Plot Threshold and Accuracy Coorelation and print best result 
    th_values=np.arange(0.0, 1.0, 0.01)
    max_acc=max(mean_value_per_threshold)
    max_acc_pos=mean_value_per_threshold.index(max_acc)
    best_result=f'Max Accuracy: {round(max_acc,3)}  at Threshold: {max_acc_pos/100} with {sampling}' 
    print(best_result)
    plt.plot(th_values, mean_value_per_threshold)
    plt.title(best_result)
    plt.suptitle('Threshold and Accuracy Coorelation')
    plt.xlabel('Threshold Value')
    plt.ylabel('Mean model accuracy for 100 different splits')
    plt.show()



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





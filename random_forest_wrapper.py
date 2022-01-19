import pandas as pd
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import data_manipulation as dm
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt

print("\n\n====== random forest ======")
rfc=RandomForestClassifier(n_estimators=500)

# we dont scale the record this time beacause chi2 method dont accept negative values (scale = 0 from data_manipulation)

for sampling in ['Oversampling', 'Undersampling']:
    X,y=dm.get_dataframe(sampling=sampling,scale="MinMax")


    mean_value_per_k=[]


    # #We try all the values for k between 1 and 15 to find witch amount of best_features has the best accuracy
    for i in range(15) :
        fvalue_selector = SelectKBest(chi2, k=(i+1))

        # Apply the SelectKBest object to the features and target
        X_kbest = fvalue_selector.fit_transform(X, y)

        result_of_split=[]

        for x in range(50):
                X_train, X_test,y_train, y_test=train_test_split(X_kbest,y,test_size=0.3,random_state=x)
                #We train the model 
                model1=rfc.fit(X_train,y_train)
                prediction1=model1.predict(X_test)
                #We save the score of eaxh training split
                result_of_split.append(accuracy_score(y_test,prediction1))
            
        #we keep only the best score from the training splits with the same k
        mean_value_per_k.append(mean(result_of_split))
        print(mean(result_of_split))

        

    max_acc=max(mean_value_per_k)

    best_result=f'Max Accuracy: {round(max_acc,3)}  with  {mean_value_per_k.index(max_acc) + 1} features and {sampling}' 
    print('\n=>',best_result)
    best_features = X.columns.values[(fvalue_selector.get_support())]
    print('Selected Features: ', best_features)

    k_values=np.arange(1, 16, 1)
    plt.plot(k_values, mean_value_per_k)
    plt.title(best_result)
    plt.suptitle('Number of Kbest Features CHI 2 method and Accuracy Coorelation')
    plt.xlabel('Features Number')
    plt.ylabel('Mean model accuracy for 100 different splits')
    plt.show()







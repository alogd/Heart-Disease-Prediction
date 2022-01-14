
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


'''
#--------------------------------------------CHI 2--------------------------------

print("\n\n CHI 2 Feature Selection \n")
lr=LogisticRegression(max_iter=1000)

# we dont scale the record this time beacause chi2 method dont accept negative values (scale = 0 from data_manipulation)
X,y=dm.get_dataframe(sampling = "oversampling", scale=0) 

mean_value_per_k=[]


# #We try all the values for k between 1 and 15 to find witch amount of best_features has the best accuracy
for i in range(15) :
    fvalue_selector = SelectKBest(chi2, k=(i+1))

    # Apply the SelectKBest object to the features and target
    X_kbest = fvalue_selector.fit_transform(X, y)

    result_of_split=[]

    for x in range(10):
            X_train, X_test,y_train, y_test=train_test_split(X_kbest,y,test_size=0.3,random_state=x)
            #We train the model 
            model1=lr.fit(X_train,y_train)
            prediction1=model1.predict(X_test)
            #We save the score of eaxh training split
            result_of_split.append(accuracy_score(y_test,prediction1))
        
    #we keep only the best score from the training splits with the same k
    mean_value_per_k.append(mean(result_of_split))
    print(mean(result_of_split))

    
#print(mean_value_per_k)

max_acc=max(mean_value_per_k)
print("max accuracy = " , max_acc , "with ", (mean_value_per_k.index(max_acc) + 1) ,  "features" )




th_values=np.arange(0, 15, 1)
max_acc=max(mean_value_per_k)
#max_acc_pos=scores_record[0].index(max_acc)
best_result=f'Max Accuracy: {round(max_acc,3)}  with {mean_value_per_k.index(max_acc) + 1} features with oversampling' 
print(best_result)
plt.plot(th_values, mean_value_per_k)
plt.title(best_result)
plt.suptitle('Number of Kbest Features CHI 2 method and Accuracy Coorelation')
plt.xlabel('Features Number')
plt.ylabel('Mean model accuracy for 100 different splits')
plt.show()

#edw exoyme thema oti bgazei kapoioa warnings sto terminal ws anafora ta max iterations oti ta ksepername....vgazei apotelesma alla prepei na doyme an epireazoun auta ta warnings




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
        for i in range(10):
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
        print(mean(result_of_split))

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

print("\n\n ANOVA Feature Selection \n")
lr=LogisticRegression()

X,y=dm.get_dataframe(sampling = "oversampling")


mean_value_per_k=[]
best_features = []
best_score = 0
# #We try all the values for k between 1 and 15 to find witch amount of best_features has the best accuracy
for i in range(15) :
    fvalue_selector = SelectKBest(f_classif, k=(i+1))

    # Apply the SelectKBest object to the features and target
    X_kbest = fvalue_selector.fit_transform(X, y)

    result_of_split=[]

    for x in range(100):
            X_train, X_test,y_train, y_test=train_test_split(X_kbest,y,test_size=0.3,random_state=x)
            #We train the model 
            model1=lr.fit(X_train,y_train)
            prediction1=model1.predict(X_test)
            #We save the score of eaxh training split
            result_of_split.append(accuracy_score(y_test,prediction1))
        
    #we keep only the best score from the training splits with the same k
    mean_value_per_k.append(mean(result_of_split))
    print(mean(result_of_split))
    
    #here we want to save the features that performed the best score 
    if mean_value_per_k[i] > best_score :
        best_score = mean_value_per_k[i]
        best_features = X.columns.values[(fvalue_selector.get_support())]


print(best_features)
    

max_acc=max(mean_value_per_k)
print("\nMax accuracy =" , max_acc , "with ", (mean_value_per_k.index(max_acc) + 1) ,  "features. These are: ", best_features  )



th_values=np.arange(0, 15, 1)
max_acc=max(mean_value_per_k)
#max_acc_pos=scores_record[0].index(max_acc)
best_result=f'\nMax Accuracy: {round(max_acc,3)}  with {mean_value_per_k.index(max_acc) + 1} features with oversampling \n\n' 
print(best_result)
plt.plot(th_values, mean_value_per_k)
plt.title(best_result)
plt.suptitle('Number of Kbest Features ANOVA method and Accuracy Coorelation')
plt.xlabel('Features Number')
plt.ylabel('Mean model accuracy for 100 different splits')
plt.show()

'''









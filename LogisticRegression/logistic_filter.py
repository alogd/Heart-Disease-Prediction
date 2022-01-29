
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import f_classif
from statistics import mean
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
import numpy as np
import sys, os
sys.path.append(os.getcwd())
import DataManipulation.data_manipulation as dm


#--------------------------------------------CHI 2--------------------------------

print("\n\n====== CHI 2 Feature Selection ======")
lr=LogisticRegression()

# we dont scale the record this time beacause chi2 method dont accept negative values (scale = 0 from data_manipulation)

for sampling in ['Oversampling', 'Undersampling']:
    X,y=dm.get_dataframe(sampling=sampling,scale="MinMax")


    mean_value_per_k=[]
    max_acc=0
    absol_max_acc=0
    # #We try all the values for k between 1 and 15 to find witch amount of best_features has the best accuracy
    for i in range(15) :
        fvalue_selector = SelectKBest(chi2, k=(i+1))

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
            
        #We keep only the best score from the training splits with the same k
        mean_value_per_k.append(mean(result_of_split))
        if max_acc<mean_value_per_k[-1]:
            max_acc=mean_value_per_k[-1]
            absol_max_acc=max(result_of_split)
        #print(mean(result_of_split))

        

    max_acc=max(mean_value_per_k)

    best_result=f'Max Accuracy: {round(max_acc,3)}  with  {mean_value_per_k.index(max_acc) + 1} features and {sampling}' 
    print('\n=>',best_result)
    print("Absolute max accuracy: ", absol_max_acc)
    best_features = X.columns.values[(fvalue_selector.get_support())]
    print('Selected Features: ', best_features)

    k_values=np.arange(1, 16, 1)
    plt.plot(k_values, mean_value_per_k)
    plt.title(best_result)
    plt.suptitle('Number of Kbest Features CHI 2 method and Accuracy Coorelation')
    plt.xlabel('Features Number')
    plt.ylabel('Mean model accuracy for 100 different splits')
    plt.show()






#---------------------VARIANCE THRESHOLD---------------------------
print("\n\n====== Variance Feature Selection ======")
lr=LogisticRegression()


for sampling in ['Oversampling', 'Undersampling']:
    X,y=dm.get_dataframe(sampling=sampling, scale="Standard")
    #We try all the values for threshold between 0 and 1 with a 0.01 step 
    currentThreshold=0
    mean_value_per_threshold=[]
    max_acc=0
    absol_max_acc=0
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
        if max_acc<mean_value_per_threshold[-1]:
            max_acc=mean_value_per_threshold[-1]
            best_features=X.columns.values[(sel.get_support())]
            absol_max_acc=max(result_of_split)

    #Plot Threshold and Accuracy Coorelation and print best result 
    th_values=np.arange(0.0, 1.0, 0.01)
    max_acc_pos=mean_value_per_threshold.index(max_acc)
    best_result=f'Max Accuracy: {round(max_acc,3)}  at Threshold: {max_acc_pos/100} with {sampling}' 
    print('\n=>',best_result)
    print('Absolute max accuracy: ', absol_max_acc)
    print('Selected Features: ', best_features)
    
    plt.plot(th_values, mean_value_per_threshold)
    plt.title(best_result)
    plt.suptitle('Threshold and Accuracy Coorelation')
    plt.xlabel('Threshold Value')
    plt.ylabel('Mean model accuracy for 100 different splits')
    plt.show()








#---------------------------------ANOVA----------------------------
print("\n\n====== ANOVA Feature Selection ======")

for sampling in ['Oversampling', 'Undersampling']:
    X,y=dm.get_dataframe(sampling=sampling, scale="Standard")
    lr=LogisticRegression()


    mean_value_per_k=[]
    best_features = []
    best_score = 0
    absol_best_score=0
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
        #print(mean(result_of_split))
        
        #here we want to save the features that performed the best score 
        if mean_value_per_k[i] > best_score :
            best_score = mean_value_per_k[i]
            best_features = X.columns.values[(fvalue_selector.get_support())]
            absol_best_score=max(result_of_split)


        

    max_acc=max(mean_value_per_k)
    best_result=f'Max Accuracy: {round(max_acc,3)}  with  {mean_value_per_k.index(max_acc) + 1} features and {sampling}' 
    print('\n=>',best_result)
    print('Absolute max accuracy: ', absol_best_score)
    print('Selected Features: ', best_features)


    k_values=np.arange(1, 16, 1)
    plt.plot(k_values, mean_value_per_k)
    plt.title(best_result)
    plt.suptitle('Number of Kbest Features ANOVA method and Accuracy Coorelation')
    plt.xlabel('Features Number')
    plt.ylabel('Mean model accuracy for 100 different splits')
    plt.show()
    












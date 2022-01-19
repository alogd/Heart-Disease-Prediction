from statistics import mean
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import data_manipulation as dm
import numpy as np
from sklearn.model_selection import train_test_split


print("\n\n====== Embedded Methods Feature Selection ======")
lr=LogisticRegression()

#Lasso
print("===>Lasso Results:")
for sampling in ['Oversampling', 'Undersampling']:
    X,y=dm.get_dataframe(sampling=sampling,scale="Standard")

    accuracy_per_C=[]
    max_acc=0
    for C in range(1,50):
        #Find best features through Lasso method
        selection = SelectFromModel(LogisticRegression(penalty='l1', C=C/50, solver='liblinear'))
        selection.fit(X, y)
        X_s=selection.transform(X)


        #Train model with the results
        result_of_split=[]
        for i in range(100):
            X_train, X_test,y_train, y_test=train_test_split(X_s,y,test_size=0.3,random_state=i)
            #print(X.columns.values[selection.get_support()])
            lr=LogisticRegression(max_iter=1000)

            model1=lr.fit(X_train,y_train)
            prediction1=model1.predict(X_test)
            #We save the score of each training split
            result_of_split.append(accuracy_score(y_test,prediction1))
        
        accuracy_per_C.append(mean(result_of_split))
        if max_acc<accuracy_per_C[-1]:
            max_acc=accuracy_per_C[-1]
            best_features=X.columns.values[selection.get_support()]

    max_acc= max(accuracy_per_C)
    C_best=(accuracy_per_C.index(max_acc)+1)/50
    best_result=f'Max accuracy: {max_acc} at C= {C_best} with {sampling}'
    print('=>',best_result)
    print("Selected Features: ",best_features)

    k_values=np.arange(0.02, 1, 0.02)
    plt.plot(k_values, accuracy_per_C)
    plt.title(best_result)
    plt.suptitle('Lasso - C and Accuracy Coorelation')
    plt.xlabel('C value')
    plt.ylabel('Mean model accuracy for 100 different splits')
    plt.show()


#Ridge
print("===>Ridge Results:")
for sampling in ['Oversampling', 'Undersampling']:
    X,y=dm.get_dataframe(sampling=sampling,scale="Standard")

    accuracy_per_C=[]
    max_acc=0
    for C in range(1,50):
        #Find best features through Lasso method
        selection = SelectFromModel(LogisticRegression(penalty='l2', C=C/50))
        selection.fit(X, y)
        X_s=selection.transform(X)


        #Train model with the results
        result_of_split=[]
        for i in range(100):
            X_train, X_test,y_train, y_test=train_test_split(X_s,y,test_size=0.3,random_state=i)
            #print(X.columns.values[selection.get_support()])
            lr=LogisticRegression(max_iter=1000)

            model1=lr.fit(X_train,y_train)
            prediction1=model1.predict(X_test)
            #We save the score of each training split
            result_of_split.append(accuracy_score(y_test,prediction1))
        
        accuracy_per_C.append(mean(result_of_split))
        if max_acc<accuracy_per_C[-1]:
            max_acc=accuracy_per_C[-1]
            best_features=X.columns.values[selection.get_support()]

    max_acc= max(accuracy_per_C)
    C_best=(accuracy_per_C.index(max_acc)+1)/50
    best_result=f'Max accuracy: {max_acc} at C= {C_best} with {sampling}'
    print('=>',best_result)
    print("Selected Features: ",best_features)

    k_values=np.arange(0.02, 1, 0.02)
    plt.plot(k_values, accuracy_per_C)
    plt.title(best_result)
    plt.suptitle('Ridge - C and Accuracy Coorelation')
    plt.xlabel('C value')
    plt.ylabel('Mean model accuracy for 100 different splits')
    plt.show()




from statistics import mean
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import sys, os
sys.path.append(os.getcwd())
import DataManipulation.data_manipulation as dm




print("\n\n====== Random Forest Plain ======\n\n")
rfc=RandomForestClassifier(n_estimators=10)


for sampling in ['Oversampling', 'Undersampling']:
    X,y=dm.get_dataframe(sampling=sampling,scale="Standard")

    result_of_split=[]

    for x in range(100):
        X_train, X_test,y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=x)
         
        model1=rfc.fit(X_train,y_train)
        prediction1=model1.predict(X_test)

        result_of_split.append(accuracy_score(y_test,prediction1))

        mean_acc = mean(result_of_split)
        max_acc= max(result_of_split)
    
    best_result=f'Max Accuracy: {round(max_acc,3)}  with  {sampling}' 
    mean_result =f'Mean Accuracy per 100 splits: {round(mean_acc,3)} with {sampling}'
    print(mean_result )
    print(best_result, "\n")









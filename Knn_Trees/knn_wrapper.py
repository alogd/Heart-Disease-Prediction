from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from statistics import mean
import sys, os
sys.path.append(os.getcwd())
import DataManipulation.data_manipulation as dm

print('\n\n====== KNN Regression Results ======')

#Create a svm Classifier
knn_model = KNeighborsClassifier()

for sampling in ['Oversampling', 'Undersampling']:
    X,y=dm.get_dataframe(sampling=sampling,scale="Standard")



    #Step Forward Feature Selection / Wrapper
    sfs1 = SFS(knn_model,
            k_features=(1,10),
            forward=True,
            floating=False,
            cv=0)

    print("\n=>Step Forward Selection + ", sampling)
    sfs1.fit(X, y)
    X_s=sfs1.transform(X)

    result_of_split=[]
    for i in range(500):
        X_train, X_test,y_train, y_test=train_test_split(X_s,y,test_size=0.3,random_state=i)
        knn_model = KNeighborsClassifier(n_neighbors=3)
        model=knn_model.fit(X_train,y_train)
        prediction=model.predict(X_test)
        result_of_split.append(accuracy_score(y_test,prediction))

    print('Mean accuracy: ',mean(result_of_split))
    print('Max accuracy: ',max(result_of_split))
    print('Selected features:',sfs1.k_feature_names_)



    #Step Backward Feature Selection / Wrapper
    sfs1 = SFS(knn_model,
            k_features=(1,10),
            forward=False,
            floating=False,
            cv=0)

    print("\n=>Step Backward Selection + ", sampling)
    sfs1.fit(X, y)
    X_s=sfs1.transform(X)

    result_of_split=[]
    for i in range(500):
        X_train, X_test,y_train, y_test=train_test_split(X_s,y,test_size=0.3,random_state=i)
        knn_model = KNeighborsClassifier(n_neighbors=3)
        model=knn_model.fit(X_train,y_train)
        prediction=model.predict(X_test)
        result_of_split.append(accuracy_score(y_test,prediction))

    print('Mean accuracy: ',mean(result_of_split))
    print('Max accuracy: ',max(result_of_split))
    print('Selected features:',sfs1.k_feature_names_)



    #Exhaustive Feature Selection / Wrapper 
    efs = EFS(knn_model,
            min_features=1,
            max_features=7,
            scoring='accuracy',
            print_progress=False,
            cv=2)
    print("\n=>Exhaustive Selection + ", sampling)

    efs.fit(X, y)
    X_s=efs.transform(X)

    result_of_split=[]
    for i in range(50):
        X_train, X_test,y_train, y_test=train_test_split(X_s,y,test_size=0.3,random_state=i)
        knn_model = KNeighborsClassifier(n_neighbors=3)
        model=knn_model.fit(X_train,y_train)
        prediction=model.predict(X_test)
        result_of_split.append(accuracy_score(y_test,prediction))

    print('Mean accuracy:',mean(result_of_split))
    print('Max accuracy: ',max(result_of_split))
    print('Selected features:',efs.best_feature_names_)


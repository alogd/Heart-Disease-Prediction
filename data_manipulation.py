import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.datasets import make_classification
from collections import Counter
StandardScaler = StandardScaler()  
ros = RandomOverSampler()
rus = RandomUnderSampler()

turn_to_dummies=['restecg']

def get_dataframe(columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'], sampling = "",scale=1):
    #Open file and read data then drop not needed columns
    df=pd.read_csv("dataset.csv")
    for col in df.columns:
        if col=='target':
            pass
        elif col not in columns:
            print("drop")
            df=df.drop([col],1)
    #turn to dummies all needed columns
    for col in columns:
        if col in turn_to_dummies:
            df=pd.get_dummies(df,columns=[col])
    
    #Scale needed columns
    if scale == 1 :
        columns_to_scale = ['age','trestbps','chol','thalach','oldpeak']

        for col in columns_to_scale:
            if col not in columns:
                columns_to_scale=columns_to_scale.remove(col)
    
        df[columns_to_scale] = StandardScaler.fit_transform(df[columns_to_scale])
    
    #Sampling dataframe
    X=df.drop(['target'], axis=1)
    y=df['target']

    X_sampled=X
    y_sampled=y

    if sampling=='oversampling':
        X_sampled, y_sampled = ros.fit_resample(X, y)
    elif sampling=='undersampling':
        X_sampled, y_sampled = rus.fit_resample(X, y)



    return X_sampled, y_sampled


        
'''
wanted_columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
X,y=get_dataframe(wanted_columns, sampling='undersampling')

print(Counter(y))
'''

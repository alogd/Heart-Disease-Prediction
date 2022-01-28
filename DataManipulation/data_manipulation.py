import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

#Making instances that may be needed
StandardScaler = StandardScaler()  
MinMaxScaler = MinMaxScaler()
ros = RandomOverSampler()
rus = RandomUnderSampler()

all_columns= ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

def get_dataframe(columns =all_columns, sampling = "",scale=""):
    '''
    This is a function that returns the desired dataset from the csv file into two variables (the independent  and dependent variables). 
    
    Parameters
    ----------
    columns: 
        Choose any subset of the 13 features. 'target' variable is always selected. 
        Default value is all features.
    
    sampling: 
        Choose `Oversampling` for RandomOverSampler or `Undersampling` for RandomUnderSampler from imblearn module. 
        Default is no sampling.

    scale: 
        Choose `Standard` for StandardScaler or `MinMax` for MinMaxScaler from sklearn module. 
        Default is no scaling.

    '''
    
    #Open file and read data then drop not needed columns
    df=pd.read_csv("DataManipulation/dataset.csv")

    for col in df.columns:
        if col=='target':
            pass
        elif col not in columns:
            df=df.drop([col],1)

    #Turn to dummies all needed columns
    turn_to_dummies=['restecg']
    for col in columns:
        if col in turn_to_dummies:
            df=pd.get_dummies(df,columns=[col])
    
    #Scale needed columns
    columns_to_scale = ['age','trestbps','chol','thalach','oldpeak']
    if scale == "Standard" :
        #Remove not needed columns
        for col in columns_to_scale:
            if col not in columns:
                columns_to_scale=columns_to_scale.remove(col)
        df[columns_to_scale] = StandardScaler.fit_transform(df[columns_to_scale])

    elif scale == "MinMax" :
        #Remove not needed columns
        for col in columns_to_scale:
            if col not in columns:
                columns_to_scale=columns_to_scale.remove(col)    
        df[columns_to_scale] = MinMaxScaler.fit_transform(df[columns_to_scale])


    
    #Sampling dataframe
    X=df.drop(['target'], axis=1)
    y=df['target']

    X_sampled=X
    y_sampled=y

    if sampling=='Oversampling':
        X_sampled, y_sampled = ros.fit_resample(X, y)
    elif sampling=='Undersampling':
        X_sampled, y_sampled = rus.fit_resample(X, y)



    return X_sampled, y_sampled



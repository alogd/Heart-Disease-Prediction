from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns

standardScaler = StandardScaler()


row_data = pd.read_csv("Dataset1.csv")



dataset = pd.get_dummies(row_data, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])



input = dataset['target']
target = dataset.drop('target', axis = 1)

linear_regression=LinearRegression().fit(input, [target])

print(linear_regression.score(input, target))



from sklearn.linear_model import LinearRegression
import pandas as pd


row_data = pd.read_csv("Dataset1.csv")

input=row_data.drop("target",1)
target=row_data["target"]

linear_regression=LinearRegression().fit(input, target)

print(linear_regression.score(input, target))



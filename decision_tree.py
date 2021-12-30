import pandas as pd
import os 
print("====" ,os.getcwd())

path='dataset.csv'
row_data = pd.read_csv(path)

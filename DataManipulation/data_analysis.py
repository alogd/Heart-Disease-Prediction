import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


df=pd.read_csv("DataManipulation/dataset.csv")

#Checking the data types
print("Data Types:")
print(df.dtypes)

#Checking the dimesnions of the dataframe
print("\n\nRows and Columns:")
print(df.shape)

#Summing the NULL values of each feature
print("\n\nNull Values:")
print(df.isnull().sum())

#Printing th first rows of the dataset
print("\n\nDataset head:")
print(df.head(3))

#Finding the correlation among the attributes
plt.figure(figsize=(20,10))
plt.title("Coorelation among attributes")
sns.heatmap(df.corr(), annot=True, cmap='PRGn')
plt.show()

#Histogram for the distribution of data
df.hist(figsize=(12,12), layout=(5,3))
plt.show()

#Box and whisker for distribution of data
df.plot(kind='box', subplots=True, layout=(5,3), figsize=(12,12))
plt.show()

#Box and whisker for the most coorelated values
#Chest pain
sns.countplot(x='cp' ,hue='target', data=df, palette='Set1')
plt.title('Chest pain')
plt.show()

#Maximum heart rate achieved
sns.lmplot(x='age',y='thalach', hue='target', data=df, palette='Set1')
plt.suptitle('Maximum heart rate achieved')
plt.show()

#Excersise induced angina
sns.countplot(x='exang' ,hue='target', data=df, palette='Set1')
plt.title('Excersise induced angina')
plt.show()


#ST depression
sns.lmplot(x='age',y='oldpeak', hue='target', data=df, palette='Set1')
plt.suptitle('ST Depression')
plt.show()


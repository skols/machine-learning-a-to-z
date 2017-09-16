# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from sklearn.preprocessing import LabelEncoder  # encoding categorical data
from sklearn.preprocessing import OneHotEncoder  # dummy encoding


# Importing the dataset
os.chdir("C:/Development/Courses/Kirill Eremenko Data Science Courses/\
Machine_Learning_A-Z/Part1_Data_Preprocessing/Section2_Data_Preprocessing/")
dataset = pd.read_csv("Data.csv")

# Create a matrix with all rows and all columns except the last one
X = dataset.iloc[:, :-1].values

# Create the dependent variable vector; last column only
y = dataset.iloc[:, 3].values

# To see the full array, run the following
np.set_printoptions(threshold=np.nan)

# Ecoding categorical data (country and purchased)
# Country first
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])  # country only

# Dummy encoding - 1 column for each country that has 1 or 0
# Column 0 = France, 1 = Germany, 2 = Spain
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

# Encode purchased
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

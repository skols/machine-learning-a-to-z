# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import Imputer  # missing data
from sklearn.preprocessing import LabelEncoder  # encoding categorical data
from sklearn.preprocessing import OneHotEncoder  # dummy encoding
# from sklearn.cross_validation import train_test_split  # deprecated
from sklearn.model_selection import train_test_split  # splitting the dataset
from sklearn.preprocessing import StandardScaler # feature scaling

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

# Taking care of missing data
# Create imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)

# Fit imputer object to missing data, which is in columns 2 and 3
# (indexes 1 and 2); upper bound is ignored, so use 3
imputer = imputer.fit(X[:, 1:3])
# Replace missing data with mean of column
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Ecoding categorical data (country and purchased)
# Country first (the independent variable)
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])  # country only

# Dummy encoding - 1 column for each country that has 1 or 0
# Column 0 = France, 1 = Germany, 2 = Spain
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

# Encode purchased (the dependant variable)
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)
# 42 is a good choice for random state

# Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# Scaling dummy variables here too
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

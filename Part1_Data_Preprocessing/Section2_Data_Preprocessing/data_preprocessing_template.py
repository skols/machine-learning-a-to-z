# Data Preprocessing

# Importing the libraries
# import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
import os
# from sklearn.cross_validation import train_test_split  # deprecated
from sklearn.model_selection import train_test_split  # splitting the dataset

# Importing the dataset
os.chdir("C:/Development/Courses/Kirill Eremenko Data Science Courses/\
Machine_Learning_A-Z/Part1_Data_Preprocessing/Section2_Data_Preprocessing/")
dataset = pd.read_csv("Data.csv")

# Create a matrix with all rows and all columns except the last one
X = dataset.iloc[:, :-1].values

# Create the dependent variable vector; last column only
y = dataset.iloc[:, 3].values

# To see the full array, run the following
# np.set_printoptions(threshold=np.nan)

# Splitting the dataset into Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)
# random_state set to 0 so we all get the same result
# 42 is a good choice for random_state otherwise

# Feature Scaling
"""
from sklearn.preprocessing import StandardScaler # feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# Scaling dummy variables here too

sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
"""

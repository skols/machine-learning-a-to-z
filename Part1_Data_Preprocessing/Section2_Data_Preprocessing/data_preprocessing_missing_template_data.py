# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import Imputer  # missing data


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

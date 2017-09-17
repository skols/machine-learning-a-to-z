# Simple linear regression

# Data Preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder  # encoding categorical data
from sklearn.preprocessing import OneHotEncoder  # dummy encoding
# from sklearn.cross_validation import train_test_split  # deprecated
from sklearn.model_selection import train_test_split  # splitting the dataset
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm  # calculate p-values and other stats


# Importing the dataset
os.chdir("C:/Development/Courses/Kirill Eremenko Data Science Courses/\
Machine_Learning_A-Z/Part2_Regression/Section5_Multiple_Linear_Regression")
dataset = pd.read_csv("50_Startups.csv")

# Create a matrix with all rows and all columns except the last one
X = dataset.iloc[:, :-1].values

# Create the dependent variable vector; last column only
y = dataset.iloc[:, 4].values

# To see the full array, run the following
# np.set_printoptions(threshold=np.nan)

# Ecoding categorical data
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])  # index 3 (state only)

# Dummy encoding - 1 column for each country that has 1 or 0
# Column 0 = California, 1 = Florida, 2 = New York
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
# Remove the first column from X
# Don't have to do manually; the library takes care of it
X = X[:, 1:]

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
# Don't need to apply feature scaling to y in this case
"""

# Fitting Multiple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
# Add a column of 1s that represents constant b0 in regression equation
# 50 lines, 1 column
# axis=1 because adding a column
# X = np.append(arr=X, values=np.ones((50, 1)).astype(int), axis=1)
# Switch X and np.ones so the 1s column is the first column
# The code is saying adding matrix of features X to an array of 1s
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)

# Create a new matrix of features that is an optimal matrix of features
# Writing the index of each column specifically because will be removing
# during backwards elimation process
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
# Ordinary Least Squares
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

regressor_OLS.summary()


# x2 has highest p-value, so it should be removed
# It is second dummy variable for state; index 2
X_opt = X[:, [0, 1, 3, 4, 5]]
# Ordinary Least Squares
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

regressor_OLS.summary()


# x1 has highest p-value now, so it should be removed
# It is first dummy variable for state; index 1
X_opt = X[:, [0, 3, 4, 5]]
# Ordinary Least Squares
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

regressor_OLS.summary()


# x2 has highest p-value now, so it should be removed
# It is Administration; index 4
X_opt = X[:, [0, 3, 5]]
# Ordinary Least Squares
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

regressor_OLS.summary()


# x2 has highest p-value now, but it's 0.06
# Going to try with it removed and compare Adjusted R-squared
# It is Marketing Spend; index 5
X_opt = X[:, [0, 3]]
# Ordinary Least Squares
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

regressor_OLS.summary()


# R-squared and Adjust R-squared are higher with Marketing Spend in,
# so I'm leaving it in
X_opt = X[:, [0, 3, 5]]
# Ordinary Least Squares
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

regressor_OLS.summary()

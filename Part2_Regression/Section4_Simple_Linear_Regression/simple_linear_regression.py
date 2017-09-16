# Simple linear regression

# Data Preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
# from sklearn.cross_validation import train_test_split  # deprecated
from sklearn.model_selection import train_test_split  # splitting the dataset
from sklearn.linear_model import LinearRegression


# Importing the dataset
os.chdir("C:/Development/Courses/Kirill Eremenko Data Science Courses/\
Machine_Learning_A-Z/Part2_Regression/Section4_Simple_Linear_Regression")
dataset = pd.read_csv("Salary_Data.csv")

# Create a matrix with all rows and all columns except the last one
X = dataset.iloc[:, :-1].values

# Create the dependent variable vector; last column only
y = dataset.iloc[:, 1].values

# To see the full array, run the following
# np.set_printoptions(threshold=np.nan)

# Splitting the dataset into Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3,
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

# Fitting Simple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
# y_pred is the vector of predictions
y_pred = regressor.predict(X_test)

# Visualizing the Training set results
plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("Salary vs. Experience (Training set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# Visualizing the Test set results
plt.scatter(X_test, y_test, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("Salary vs. Experience (Test set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

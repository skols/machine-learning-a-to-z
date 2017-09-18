# SVR

# Data Preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler # feature scaling
# from sklearn.cross_validation import train_test_split  # deprecated
# from sklearn.model_selection import train_test_split  # splitting the dataset
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
# import statsmodels.formula.api as sm  # calculate p-values and other stats


# Importing the dataset
os.chdir("C:/Development/Courses/Kirill Eremenko Data Science Courses/\
Machine_Learning_A-Z/Part2_Regression/Section7_SVR")
dataset = pd.read_csv("Position_Salaries.csv")

# Create a matrix with all rows and the Level column only
X = dataset.iloc[:, 1:2].values

# Create the dependent variable vector; last column only
y = dataset.iloc[:, 2].values

# To see the full array, run the following
# np.set_printoptions(threshold=np.nan)

# Splitting the dataset into Training set and Test set - not creating here
# Only 10 rows so wouldn't provide value
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)
"""
# random_state set to 0 so we all get the same result
# 42 is a good choice for random_state otherwise

# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


# Fitting the SVR to the dataset
# Create your regressor here
regressor = SVR(kernel="rbf")
regressor.fit(X, y)

# Predicting a new result with SVR
# 2 pairs of brackets means an array of 1 line and 1 cell
y_pred = sc_y.inverse_transform(regressor.
                                predict(sc_X.transform(np.array([[6.5]]))))

# Visualising the SVR results
plt.scatter(X, y, color="red")
plt.plot(X, regressor.predict(X), color="blue")
plt.title("Truth or Bluff (SVR)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color="red")
plt.plot(X_grid, regressor.predict(X_grid), color="blue")
plt.title("Truth or Bluff (SVR)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Polynomial Regression

# Data Preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
# from sklearn.cross_validation import train_test_split  # deprecated
# from sklearn.model_selection import train_test_split  # splitting the dataset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.formula.api as sm  # calculate p-values and other stats


# Importing the dataset
os.chdir("C:/Development/Courses/Kirill Eremenko Data Science Courses/\
Machine_Learning_A-Z/Part2_Regression/Section6_Polynomial_Regression")
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
"""
from sklearn.preprocessing import StandardScaler # feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# Scaling dummy variables here too
# Don't need to apply feature scaling to y in this case
"""

# Fitting Linear Regression to the dataset
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
poly_reg = PolynomialFeatures(degree=4)
# Changing degree from 2 to 4 makes prediction more accurate
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color="red")
plt.plot(X, lin_reg.predict(X), color="blue")
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Visualising the Polynomial Regression results
# Original
# plt.scatter(X, y, color="red")
# plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color="blue")
# plt.title("Truth or Bluff (Polynomial Regression)")
# plt.xlabel("Position Level")
# plt.ylabel("Salary")
# plt.show()

# Using X_grid for a smoother curve that increments by 0.1
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color="red")
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)),
         color="blue")
plt.title("Truth or Bluff (Polynomial Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Predicting a new result with Linear Regression
# 6.5 level
lin_reg.predict(6.5)

# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))

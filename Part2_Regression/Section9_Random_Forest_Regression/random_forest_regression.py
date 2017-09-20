# Random Forest Regression - Non-continuous model

# Data Preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
# from sklearn.cross_validation import train_test_split  # deprecated
# from sklearn.model_selection import train_test_split  # splitting the dataset
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures
# import statsmodels.formula.api as sm  # calculate p-values and other stats
from sklearn.ensemble import RandomForestRegressor

# Importing the dataset
os.chdir("C:/Development/Courses/Kirill Eremenko Data Science Courses/\
Machine_Learning_A-Z/Part2_Regression/Section9_Random_Forest_Regression")
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

# Fitting the Random Forest Regression model to the dataset
# regressor = RandomForestRegressor(n_estimators=10, random_state=0)
# regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(X, y)

# Predicting a new result with the Random Forest Regression Model
y_pred = regressor.predict(6.5) # 160333 with 300

# Visualising the Random Forest Regression results (for higher resolution
# and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color="red")
plt.plot(X_grid, regressor.predict(X_grid), color="blue")
plt.title("Truth or Bluff (Random Forest Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

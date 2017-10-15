# XGBoost

# Install xgboost following the instructions on this link:
# http://xgboost.readthedocs.io/en/latest/build.html#

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder  # encoding categorical data
from sklearn.preprocessing import OneHotEncoder  # dummy encoding
# from sklearn.cross_validation import train_test_split  # deprecated
from sklearn.model_selection import train_test_split  # splitting the dataset
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score  # k-Fold Cross Validation


# To see the full array, run the following
np.set_printoptions(threshold=np.nan)

# Importing the dataset
os.chdir("C:/Development/Courses/Kirill Eremenko Data Science Courses/\
Machine_Learning_A-Z/Part10_Model_Selection_And_Boosting/\
Section39_XGBoost")
dataset = pd.read_csv("Churn_Modelling.csv")

# Create a matrix of features
X = dataset.iloc[:, 3:13].values  # 13 because upper bound excluded in range

# Create the dependent variable vector; last column only
y = dataset.iloc[:, 13].values

# Ecoding categorical data (country and gender)
# Country first (the independent variable)
labelencoder_X_1 = LabelEncoder()  # country
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])  # country only
labelencoder_X_2 = LabelEncoder()  # gender
X[:, 2] = labelencoder_X_1.fit_transform(X[:, 2])  # country only

# Dummy encoding - 1 column for each country that has 1 or 0
# Column 0 = France, 1 = Germany, 2 = Spain
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()

# Remove first country dummy variable to avoid dummy variable trap
X = X[:, 1:]

# Splitting the dataset into Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)
# random_state set to 0 so we all get the same result
# 42 is a good choice for random_state otherwise

# Fitting XGBoost to the Training set
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
# 10 elements with 10 accuracies to evaluate the model
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train,
                             cv=10)
accuracies.mean()
accuracies.std()

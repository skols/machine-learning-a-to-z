# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+https://github.com/Theano/Theano.git#egg=Theano

# Install Tensorflow
# https://www.tensorflow.org/install/install_windows
# Using anaconda, so follow the steps below
    # conda create -n tensorflow python=3.6
    # activate tensorflow
    # Installing CPU version for this
        # CPU version: pip install --upgrade tensorflow
        # GPU version: pip install --upgrade tensorflow-gpu
    
    # deactivate
    
# The above didn't work, so run this from Anaconda Prompt as administrator
# pip install --upgrade tensorflow

# Keras is a library that runs on Tensorflow and Theano
# Makes our code simpler

# Installing Keras (use Anaconda Prompt as administrator)
# pip install --upgrade keras

# Part 1 - Data Preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder  # encoding categorical data
from sklearn.preprocessing import OneHotEncoder  # dummy encoding
# from sklearn.cross_validation import train_test_split  # deprecated
from sklearn.model_selection import train_test_split  # splitting the dataset
from sklearn.preprocessing import StandardScaler # feature scaling
from sklearn.metrics import confusion_matrix

# To see the full array, run the following
np.set_printoptions(threshold=np.nan)

# Importing the dataset
os.chdir("C:/Development/Courses/Kirill Eremenko Data Science Courses/\
Machine_Learning_A-Z/Part8_Deep_Learning/\
Section31_Artificial_Neural_Networks")
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

# Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Part 2 - Make the ANN
import keras
from keras.models import Sequential  # initialize the ANN
from keras.layers import Dense  # used to create the layers in the ANN

# Initializing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
# units is nodes in hidden layer; 6 is average of inputs and outputs (11+1)/2
# Updated Dense documentation doesn't show input_dim but still needed
# relu is for Rectifier Activation Function for hidden layers
classifier.add(Dense(units=6, kernel_initializer="uniform",
                     activation="relu", input_dim=11))

# Adding the second hidden layer
classifier.add(Dense(units=6, kernel_initializer="uniform",
                     activation="relu"))

# Adding the output layer
# 1 node on output layer because binary outcome
# sigmoid is for Sigmoid Activation Function for output layer
# softmax is for Sigmoid Activation Function for output layer with more than 2
# categories
classifier.add(Dense(units=1, kernel_initializer="uniform",
                     activation="sigmoid"))

# Compiliing the ANN
# adam is a Stochastic Gradient Descent algorith (one of several)
# Check Keras documentation to learn more about optimizer, loss, and metrics
classifier.compile(optimizer="adam", loss="binary_crossentropy",
                   metrics=["accuracy"])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Part 3 - Making the predictions and evaluating the model
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# If a probability is greater than 0.5, True, else False
y_pred = (y_pred > 0.5)  # Convert predictions to True or False

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
(1493+223)/2000

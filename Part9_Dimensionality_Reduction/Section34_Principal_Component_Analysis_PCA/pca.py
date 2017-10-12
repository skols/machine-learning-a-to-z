# PCA - Principal Component Analysis

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
# from sklearn.cross_validation import train_test_split  # deprecated
from sklearn.model_selection import train_test_split  # splitting the dataset
from sklearn.preprocessing import StandardScaler # feature scaling
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap  # prediction regions plot
from sklearn.decomposition import PCA


# To see the full array, run the following
np.set_printoptions(threshold=np.nan)

# Importing the dataset
os.chdir("C:/Development/Courses/Kirill Eremenko Data Science Courses/\
Machine_Learning_A-Z/Part9_Dimensionality_Reduction/\
Section34_Principal_Component_Analysis_PCA")
dataset = pd.read_csv("Wine.csv")

# Create a matrix of features with Age and EstimatedSalary
X = dataset.iloc[:, :13].values

# Create the dependent variable vector; last column only
y = dataset.iloc[:, 13].values

# Splitting the dataset into Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)
# random_state set to 0 so we all get the same result
# 42 is a good choice for random_state otherwise

# Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Applying PCA
# pca = PCA(n_components=None)  # Do this first to see how many are needed
# X_train = pca.fit_transform(X_train)
# use transform, not fit_transform, because already fitted to the training set
# X_test = pca.transform(X_test)
# explained_variance = pca.explained_variance_ratio_
# The top 2 explain 56% of the variance, which is good to make a
# classification model
pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
# use transform, not fit_transform, because already fitted to the training set
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

# Fitting Logistic Regression to the Training set
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
# Adding a third color, blue, because of 3 classes
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1,
                               stop=X_set[:, 0].max() + 1,
                               step = 0.01),
                     np.arange(start=X_set[:, 1].min() - 1,
                               stop=X_set[:, 1].max() + 1,
                               step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).
             reshape(X1.shape), alpha=0.75, cmap=ListedColormap(("red",
                    "green", "blue")))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', "blue"))(i), label = j)
plt.title("Logistic Regression (Training Set)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.show()


# Visualising the Test set results
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1,
                               stop=X_set[:, 0].max() + 1,
                               step = 0.01),
                     np.arange(start=X_set[:, 1].min() - 1,
                               stop=X_set[:, 1].max() + 1,
                               step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).
             reshape(X1.shape), alpha=0.75, cmap=ListedColormap(("red",
                    "green", "blue")))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', "blue"))(i), label = j)
plt.title("Logistic Regression (Test Set)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.show()

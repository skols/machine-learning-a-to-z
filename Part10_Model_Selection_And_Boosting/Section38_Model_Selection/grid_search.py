# Grid Search

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
# from sklearn.cross_validation import train_test_split  # deprecated
from sklearn.model_selection import train_test_split  # splitting the dataset
from sklearn.preprocessing import StandardScaler # feature scaling
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap  # prediction regions plot
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score  # k-Fold Cross Validation
from sklearn.model_selection import GridSearchCV


# To see the full array, run the following
np.set_printoptions(threshold=np.nan)

# Importing the dataset
os.chdir("C:/Development/Courses/Kirill Eremenko Data Science Courses/\
Machine_Learning_A-Z/Part10_Model_Selection_And_Boosting/\
Section38_Model_Selection")
dataset = pd.read_csv("Social_Network_Ads.csv")

# Create a matrix of features with Age and EstimatedSalary
X = dataset.iloc[:, [2, 3]].values

# Create the dependent variable vector; last column only
y = dataset.iloc[:, 4].values

# To see the full array, run the following
# np.set_printoptions(threshold=np.nan)

# Splitting the dataset into Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=0)
# random_state set to 0 so we all get the same result
# 42 is a good choice for random_state otherwise

# Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting Kernel SVM to the Training set
classifier = SVC(kernel="rbf", random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
# 10 elements with 10 accuracies to evaluate the model
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train,
                             cv=10)
accuracies.mean()  # 0.9005
# 90% is the relevant evaluation of model performance
accuracies.std()  # 0.0638

# Applying Grid Search to find the best model and best parameters
parameters = [{"C": [1, 10, 100, 1000], "kernel": ["linear"]},
               {"C": [1, 10, 100, 1000], "kernel": ["rbf", "sigmoid", "poly"],
                "gamma": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                "degree": [2, 3, 4, 5, 6]},
               # .5 because default is 1/n_features; 2 features here, so 1/2
        ]
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters,
                           scoring="accuracy", cv=10, n_jobs=-1)
# n_jobs=-1 uses all CPUs
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

# Visualising the Training set results
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1,
                               stop=X_set[:, 0].max() + 1,
                               step = 0.01),
                     np.arange(start=X_set[:, 1].min() - 1,
                               stop=X_set[:, 1].max() + 1,
                               step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).
             reshape(X1.shape), alpha=0.75, cmap=ListedColormap(("red",
                    "green")))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title("Kernel SVM (Training Set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
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
                    "green")))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title("Kernel SVM (Test Set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()

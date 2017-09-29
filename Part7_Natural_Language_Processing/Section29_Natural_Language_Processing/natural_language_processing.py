# template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split  # splitting the dataset
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB


# To see the full array, run the following
np.set_printoptions(threshold=np.nan)

# Importing the dataset
os.chdir("C:/Development/Courses/Kirill Eremenko Data Science Courses/\
Machine_Learning_A-Z/Part7_Natural_Language_Processing/\
Section29_Natural_Language_Processing")
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter="\t",
                      quoting=3) # quoting=3 ignores double quotes in file

"""
Cleaning the text
Remove non-significant words, puncuation, numbers
Applying stemming to get root of a word, e.g. loved to love
Stemming is done to avoid too much sparsity in the sparse matrix that will
be created
Get rid of capital letters


# To remove puncuation and number, select what we want to keep
review = re.sub("[^a-zA-Z]", " ", dataset["Review"][0])
# Convert all letters to lowercase
review = review.lower()
# Remove non-significant words - and, that, this, in, of, and so on
review = review.split()  # Convert review from str to list of its 4 words
# Stemming
ps = PorterStemmer()
review = [ps.stem(word) for word in review if not word in\
          set(stopwords.words("english"))]
# algorithm can be faster when it's a set instead of a list

# Convert review back to a string
review = " ".join(review)
"""

# Apply cleaning to entire dataset
corpus = []
rows = dataset.shape[0]
for i in range(0, rows):
    review = re.sub("[^a-zA-Z]", " ", dataset["Review"][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in\
              set(stopwords.words("english"))]
    review = " ".join(review)
    corpus.append(review)

# Creating the Bag of Words model
"""
Table where each row is a review and each column is each word from all the
reviews - sparse matrix (a lot of 0s is called sparsity)
Then can use a classification model to predict if review is positive or
negative
"""
cv = CountVectorizer(max_features=1500)  # Just keep 1500 most frequent words
X = cv.fit_transform(corpus).toarray()  # toarray() to make X a matrix
y = dataset.iloc[:, 1].values

# Splitting the dataset into Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                    random_state=0)
# random_state set to 0 so we all get the same result
# 42 is a good choice for random_state otherwise

def cm_stats(y_test, y_pred, class_type):
    cm = confusion_matrix(y_test, y_pred)
    accuracy = (cm[0][0] + cm[1][1])/len(y_test)
    precision = cm[1][1]/(cm[1][1] + cm[0][1])
    recall = cm[1][1]/(cm[1][1] + cm[1][0])
    f1_score = 2 * precision * recall / (precision + recall)
    print("{0}\n cm: {1}, accuracy: {2}, precision: {3}, recall: {4},\
 f1_score: {5}".format(class_type, cm, accuracy, precision, recall, f1_score))

# Fitting Naive Bayes to the Training set
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

cm_stats(y_test, y_pred, "Naive Bayes")

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

cm_stats(y_test, y_pred, "Logistic Regression")


# Fitting K-NN to the Training set
# metric="minkowski" and p=2 so Euclidean distance is used
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

cm_stats(y_test, y_pred, "K-NN")


# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel="linear", random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

cm_stats(y_test, y_pred, "SVM")


# Fitting Kernel SVM to the Training set
classifier = SVC(kernel="rbf", random_state=42)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

cm_stats(y_test, y_pred, "Kernel SVM")


# Fitting Decision Tree to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion="entropy", random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

cm_stats(y_test, y_pred, "Decision Tree")


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10,
                                    criterion="entropy",
                                    random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

cm_stats(y_test, y_pred, "Random Forest")


# Fitting CART to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

cm_stats(y_test, y_pred, "CART")


# Fitting Maximum Entropy to the Training set
from nltk.classify.maxent import MaxentClassifier
algorithm = nltk.classify.MaxentClassifier.ALGORITHMS[0]
classifier = MaxentClassifier.train()


# Predicting the Test set results
y_pred = classifier.predict(X_test)

cm_stats(y_test, y_pred, "CART")

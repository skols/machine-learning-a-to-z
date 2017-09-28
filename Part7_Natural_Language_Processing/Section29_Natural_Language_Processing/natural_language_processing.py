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

# Fitting Naive Bayes to the Training set
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

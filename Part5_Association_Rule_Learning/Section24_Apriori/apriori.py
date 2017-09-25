# Apriori

# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
# from sklearn.cross_validation import train_test_split  # deprecated
from apyori import apriori


# Importing the dataset
os.chdir("C:/Development/Courses/Kirill Eremenko Data Science Courses/\
Machine_Learning_A-Z/Part5_AssociationRuleLearning/Section24_Apriori")
dataset = pd.read_csv("Market_Basket_Optimisation.csv", header=None)

# To see the full array, run the following
np.set_printoptions(threshold=np.nan)

# Need to create a list of lists that has the transactions
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0,20)])

# Training Apriori on the dataset
# Minimum Support: Products purchased at least 3 times a day; 7 days a week;
# 7500 total transactions
# (3*7)/7500=0.0028
rules = apriori(transactions, min_support=0.003, min_confidence=0.2,
                min_lift=3, min_length=2)

# Visualising the results
results = list(rules)
results[0:5]

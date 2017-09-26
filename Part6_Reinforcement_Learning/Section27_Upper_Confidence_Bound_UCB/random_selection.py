# Random Selection

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import os
# from sklearn.cross_validation import train_test_split  # deprecated
# from sklearn.model_selection import train_test_split  # splitting the dataset


# To see the full array, run the following
np.set_printoptions(threshold=np.nan)

# Importing the dataset
os.chdir("C:/Development/Courses/Kirill Eremenko Data Science Courses/\
Machine_Learning_A-Z/Part6_Reinforcement_Learning/\
Section27_Upper_Confidence_Bound_UCB")
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

# Implementing Random Selection
N = 10000
d = 10
ads_selected = []
total_reward = 0
for n in range(0, N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_reward = total_reward + reward

# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

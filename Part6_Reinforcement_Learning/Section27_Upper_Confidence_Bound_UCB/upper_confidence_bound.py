# Upper Confidence Bound

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from math import sqrt, log

# To see the full array, run the following
np.set_printoptions(threshold=np.nan)

# Importing the dataset
os.chdir("C:/Development/Courses/Kirill Eremenko Data Science Courses/\
Machine_Learning_A-Z/Part6_Reinforcement_Learning/\
Section27_Upper_Confidence_Bound_UCB")
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

# Find the ad that'll get the most clicks
# 10 versions of the same ad
# Each time a user logins, he/she will see one ad
# If he/she clicks on the ad, it gets a 1; if he/she doesn't, it gets a 0
# Strategy on what ad is shown depends on what previous user did

# Implementing UCB
# Step 1
N = len(dataset)
d = 10  # number of ads
ads_selected = []
numbers_of_selections = [0] * d  # vector of size d containing only 0s
sums_of_rewards = [0] * d
total_reward = 0

# Step 2
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if numbers_of_selections[i] > 0:
            # The 10 first rounds select the 10 ads and after round 10, the
            # below strategy gets used
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = sqrt((3/2) * (log(n+1)/numbers_of_selections[i]))
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        # Step 3
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i  # Keeping track of the ad that has the max
    ads_selected.append(ad)
#   numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    numbers_of_selections[ad] += 1
    reward = dataset.values[n, ad]
#   sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    sums_of_rewards[ad] += reward
    total_reward += reward

# Visualising the results
plt.hist(ads_selected)
plt.title("Histogram of ads selections")
plt.xlabel("Ads")
plt.ylabel("Number of times each ad was selected")
plt.show()

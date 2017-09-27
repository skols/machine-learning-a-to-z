# Thompson Sampling

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from random import betavariate


# To see the full array, run the following
np.set_printoptions(threshold=np.nan)

# Importing the dataset
os.chdir("C:/Development/Courses/Kirill Eremenko Data Science Courses/\
Machine_Learning_A-Z/Part6_Reinforcement_Learning/\
Section28_Thompson_Sampling")
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

# Implementing Thompson Sampling
# Step 1
N = len(dataset)
d = 10  # number of ads
ads_selected = []
number_of_rewards_1 = [0] * d
number_of_rewards_0 = [0] * d
total_reward = 0

# Step 2
for n in range(0, N):
    ad = 0
    max_random_draw = 0
    for i in range(0, d):
        random_beta = betavariate(number_of_rewards_1[i] + 1,
                                         number_of_rewards_0[i] + 1)
        # Step 3
        if random_beta > max_random_draw:
            max_random_draw = random_beta
            ad = i  # Keeping track of the ad that has the max
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        number_of_rewards_1[ad] += 1
    else:
        number_of_rewards_0[ad] += 1
    total_reward += reward

# Visualising the results
plt.hist(ads_selected)
plt.title("Histogram of ads selections")
plt.xlabel("Ads")
plt.ylabel("Number of times each ad was selected")
plt.show()

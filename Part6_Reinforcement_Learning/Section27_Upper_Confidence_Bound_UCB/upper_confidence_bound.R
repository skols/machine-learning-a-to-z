# Upper Confidence Bound

# Importing the dataset
setwd("C:/Development/Courses/Kirill Eremenko Data Science Courses/Machine_Learning_A-Z/Part6_Reinforcement_Learning/Section27_Upper_Confidence_Bound_UCB")
dataset <- read.csv("Ads_CTR_Optimisation.csv")

# Implementing UCB
# Step 1
N <- nrow(dataset)  # 10000
d <- length(dataset)  # 10
ads_selected <- integer(0)
number_of_selections <- integer(d)
sums_of_rewards <- integer(d)
total_reward <- 0

# Step 2
for (n in 1:N) {
  ad <- 0
  max_upper_bound <- 0
  for (i in 1:d) {
    if (number_of_selections[i] > 0) {
      # The 10 first rounds select the 10 ads and after round 10, the below strategy gets used
      average_reward <- sums_of_rewards[i]/number_of_selections[i]
      delta_i = sqrt((3/2) * (log(n)/number_of_selections[i]))
      upper_bound = average_reward + delta_i
    }
    else {
      upper_bound <- 1e400
    }
    # Step 3
    if (upper_bound > max_upper_bound) {
      max_upper_bound <- upper_bound
      ad <- i
    }
  }
  ads_selected <- append(ads_selected, ad)
  number_of_selections[ad] <- number_of_selections[ad] + 1
  reward <- dataset[n, ad]  # Actual record from the dataset
  sums_of_rewards[ad] <- sums_of_rewards[ad] + reward
  total_reward <- total_reward + reward
}

# Visualising the results
hist(ads_selected,
     col = 'blue',
     main = 'Histogram of ads selections',
     xlab = 'Ads',
     ylab = 'Number of times each ad was selected')

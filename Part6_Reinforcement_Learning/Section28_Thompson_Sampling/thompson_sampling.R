# Thompson Sampling

# Importing the dataset
setwd("C:/Development/Courses/Kirill Eremenko Data Science Courses/Machine_Learning_A-Z/Part6_Reinforcement_Learning/Section28_Thompson_Sampling")
dataset <- read.csv("Ads_CTR_Optimisation.csv")

# Implementing Thompson Sampling
# Step 1
N <- nrow(dataset)  # 10000
d <- length(dataset)  # 10
ads_selected <- integer(0)
number_of_rewards_1 <- integer(d)
number_of_rewards_0 <- integer(d)
total_reward <- 0

# Step 2
for (n in 1:N) {
  ad <- 0
  max_random_draw <- 0
  for (i in 1:d) {
    random_beta <- rbeta(n=1,
                         shape1=number_of_rewards_1[i] + 1,
                         shape2=number_of_rewards_0[i] + 1)
    # Step 3
    if (random_beta > max_random_draw) {
      max_random_draw <- random_beta
      ad <- i
    }
  }
  ads_selected <- append(ads_selected, ad)
  reward <- dataset[n, ad]  # Actual record from the dataset
  if (reward == 1) {
    number_of_rewards_1[ad] <- number_of_rewards_1[ad] + 1
  } else {
    number_of_rewards_0[ad] <- number_of_rewards_0[ad] + 1
  }
  total_reward <- total_reward + reward
}

# Visualising the results
hist(ads_selected,
     col = 'blue',
     main = 'Histogram of ads selections',
     xlab = 'Ads',
     ylab = 'Number of times each ad was selected')

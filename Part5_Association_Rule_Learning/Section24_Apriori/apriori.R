# Apriori

# Data Preprocessing

# Importing the dataset
setwd("C:/Development/Courses/Kirill Eremenko Data Science Courses/Machine_Learning_A-Z/Part5_AssociationRuleLearning/Section24_Apriori")
dataset <- read.csv("Market_Basket_Optimisation.csv", header=FALSE)

# Create a sparse matrix with 120 columns (one for each item) and 0 or 1 in the records (1 if bought, 0 if not)
# install.packages("arules")
library(arules)
dataset <- read.transactions("Market_Basket_Optimisation.csv", sep=",", rm.duplicates=TRUE)
summary(dataset)

itemFrequencyPlot(dataset, topN=10)

# Training Apriori on the dataset
# Minimum Support: Products purchased at least 3 times a day; 7 days a week; 7500 total transactions
# (3*7)/7500=0.0028
rules <- apriori(data=dataset, parameter=list(support=0.003, confidence=0.8))  # 0 rules because confidence too high
rules <- apriori(data=dataset, parameter=list(support=0.003, confidence=0.4))
rules <- apriori(data=dataset, parameter=list(support=0.003, confidence=0.2))
# (4*7)/7500=0.00373333
rules <- apriori(data=dataset, parameter=list(support=0.004, confidence=0.2))

# Visualising the results
inspect(sort(rules, by="lift")[1:10])

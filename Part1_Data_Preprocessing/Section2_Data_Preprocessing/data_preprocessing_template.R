# Data Preprocessing

# Importing the dataset
setwd("C:/Development/Courses/Kirill Eremenko Data Science Courses/Machine_Learning_A-Z/Part1_Data_Preprocessing/Section2_Data_Preprocessing")
dataset <- read.csv("Data.csv")

# Splitting the dataset into Training set and Test set
library(caTools)
set.seed(123)  # like random_state
split <- sample.split(dataset$Purchased, SplitRatio=0.8)  # for training set
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

# Feature Scaling
# training_set[, 2:3] <- scale(training_set[, 2:3])  # Excluding categories since they're factors
# test_set[, 2:3] <- scale(test_set[, 2:3])

# Simple linear regression
# Data Preprocessing

# Importing the dataset
setwd("C:/Development/Courses/Kirill Eremenko Data Science Courses/Machine_Learning_A-Z/Part2_Regression/Section4_Simple_Linear_Regression")
dataset = read.csv("Salaray_Data.csv")
# dataset = dataset[, 2:3]

# Splitting the dataset into Training set and Test set
library(caTools)
set.seed(123)  # like random_state
split = sample.split(dataset$Purchased, SplitRatio=2/3)  # for training set
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set[, 2:3] = scale(training_set[, 2:3])  # Excluding categories since they're factors
# test_set[, 2:3] = scale(test_set[, 2:3])

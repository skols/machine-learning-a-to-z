# Data Preprocessing

# Importing the dataset
setwd("C:/Development/Courses/Kirill Eremenko Data Science Courses/Machine_Learning_A-Z/Part1_Data_Preprocessing/Section2_Data_Preprocessing")
dataset <- read.csv("Data.csv")

# Take care of missing data
dataset$Age <- ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN=function(x) mean(x, na.rm=TRUE)),
                     dataset$Age)

dataset$Salary <- ifelse(is.na(dataset$Salary),
                     ave(dataset$Salary, FUN=function(x) mean(x, na.rm=TRUE)),
                     dataset$Salary)

# Encoding categorical data
# Transform country to a column of factors
dataset$Country <- factor(dataset$Country,
                         levels=c("France", "Spain", "Germany"),
                         labels=c(1, 2, 3))

# Transform purchased to a factor
dataset$Purchased <- factor(dataset$Purchased,
                         levels=c("No", "Yes"),
                         labels=c(0, 1))

# Splitting the dataset into Training set and Test set
library(caTools)
set.seed(123)  # like random_state
split <- sample.split(dataset$Purchased, SplitRatio=0.8)  # for training set
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

# Feature Scaling
training_set[, 2:3] <- scale(training_set[, 2:3])  # Excluding categories since they're factors
test_set[, 2:3] <- scale(test_set[, 2:3])

# XGBoost

# Importing the dataset
setwd("C:/Development/Courses/Kirill Eremenko Data Science Courses/Machine_Learning_A-Z/Part10_Model_Selection_And_Boosting/Section39_XGBoost")
dataset <- read.csv("Churn_Modelling.csv")
dataset <- dataset[4:14]

# Encoding the categorical data as factors
# Have to be numeric because of model we'll be using
# Transform Geography to a column of factors
dataset$Geography <- as.numeric(factor(dataset$Geography,
                                       levels=c("France", "Spain", "Germany"),
                                       labels=c(1, 2, 3)))

# Transform Gender to a factor
dataset$Gender <- as.numeric(factor(dataset$Gender,
                                    levels=c("Female", "Male"),
                                    labels=c(1, 2)))

# Splitting the dataset into Training set and Test set
library(caTools)
set.seed(123)  # like random_state
split <- sample.split(dataset$Exited, SplitRatio=0.8)  # for training set
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

# Fitting XGBoost to the Training set
# install.packages("xgboost")
library(xgboost)
# training_set$Exited is dependent variable as a vector
classifier <- xgboost(data=as.matrix(training_set[-11]), label=training_set$Exited,
                      nrounds=10)

# Predicting the Test set results
y_pred <- predict(classifier, newdata=as.matrix(test_set[-11]))
y_pred <- (y_pred >= 0.5)

# Making the Confusion Matrix
cm = table(test_set[, 11], y_pred)

# Applying k-Fold Cross Validation
library(caret)
folds <- createFolds(training_set$Exited, k=10)
cv <- lapply(folds, function(x) {  # x is each element of folds
  training_fold <- training_set[-x, ]
  test_fold <- training_set[x, ]
  classifier <- xgboost(data=as.matrix(training_set[-11]), label=training_set$Exited,
                        nrounds=10)
  y_pred <- predict(classifier, newdata=as.matrix(test_fold[-11]))
  y_pred <- (y_pred >= 0.5)
  cm <- table(test_fold[, 11], y_pred)
  accuracy <- (cm[1,1] + cm[2,2])/(cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy)
})
accuracy <- mean(as.numeric(cv))

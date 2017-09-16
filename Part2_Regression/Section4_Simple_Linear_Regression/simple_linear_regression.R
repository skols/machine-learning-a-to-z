# Simple linear regression
# Data Preprocessing

# Importing the dataset
setwd("C:/Development/Courses/Kirill Eremenko Data Science Courses/Machine_Learning_A-Z/Part2_Regression/Section4_Simple_Linear_Regression")
dataset <- read.csv("Salary_Data.csv")
# dataset <- dataset[, 2:3]

# Splitting the dataset into Training set and Test set
library(caTools)
set.seed(123)  # like random_state
split <- sample.split(dataset$Salary, SplitRatio=2/3)  # for training set
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

# Feature Scaling
# training_set[, 2:3] <- scale(training_set[, 2:3])  # Excluding categories since they're factors
# test_set[, 2:3] <- scale(test_set[, 2:3])

# Fitting Simple Linear Regression to the Training set
regressor <- lm(formula=Salary ~ YearsExperience, data=training_set)

# Predicting the Test set results
# y_pred is the vector that will contain the predicted values
y_pred <- predict(regressor, newdata=test_set)

# Visualising the Training set results
library(ggplot2)
ggplot() +
  geom_point(aes(x=training_set$YearsExperience, y=training_set$Salary),
             color="red") +
  geom_line(aes(x=training_set$YearsExperience, y=predict(regressor, newdata=training_set)),
            color="blue") +
  ggtitle("Salary vs. Experience (Training set)") +
  xlab("Years of Experience") +
  ylab("Salary")

# Visualising the Test set results
ggplot() +
  geom_point(aes(x=test_set$YearsExperience, y=test_set$Salary),
             color="red") +
  geom_line(aes(x=training_set$YearsExperience, y=predict(regressor, newdata=training_set)),
            color="blue") +
  ggtitle("Salary vs. Experience (Test set)") +
  xlab("Years of Experience") +
  ylab("Salary")

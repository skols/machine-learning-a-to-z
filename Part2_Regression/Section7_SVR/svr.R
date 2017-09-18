# SVR

# Loading libraries
library(ggplot2)

# Importing the dataset
setwd("C:/Development/Courses/Kirill Eremenko Data Science Courses/Machine_Learning_A-Z/Part2_Regression/Section7_SVR")
dataset <- read.csv("Position_Salaries.csv")
dataset <- dataset[, 2:3]

# Splitting the dataset into Training set and Test set
# library(caTools)
# set.seed(123)  # like random_state
# split <- sample.split(dataset$Profit, SplitRatio=0.8)  # for training set
# training_set <- subset(dataset, split == TRUE)
# test_set <- subset(dataset, split == FALSE)

# Feature Scaling
# training_set[, 2:3] <- scale(training_set[, 2:3])  # Excluding categories since they're factors
# test_set[, 2:3] <- scale(test_set[, 2:3])

# Fitting SVR to the dataset
library(e1071)  # includes feature scaling
regressor <- svm(formula=Salary ~ ., data=dataset, type="eps-regression")

# Predicting a new result
y_pred <- predict(regressor, data.frame(Level=6.5))

# Visualising the SVR results
ggplot() +
  geom_point(aes(x=dataset$Level, y=dataset$Salary), color="red") +
  geom_line(aes(x=dataset$Level, y=predict(regressor, newdata=dataset)),
            color="blue") +
  ggtitle("Truth or Bluff (SVR)") +
  xlab("Level") +
  ylab("Salary")


# Visualising the SVR results
# Higher resolution and smoother curve
x_grid <- seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x=dataset$Level, y=dataset$Salary), color="red") +
  geom_line(aes(x=x_grid, y=predict(regressor, newdata=data.frame(Level=x_grid))),
            color="blue") +
  ggtitle("Truth or Bluff (SVR)") +
  xlab("Level") +
  ylab("Salary")

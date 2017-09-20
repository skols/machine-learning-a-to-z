# Random Forest Regression - Non-continuous

# Loading libraries
library(ggplot2)
library(randomForest)

# Importing the dataset
setwd("C:/Development/Courses/Kirill Eremenko Data Science Courses/Machine_Learning_A-Z/Part2_Regression/Section9_Random_Forest_Regression")
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

# Fitting Random Forest Regression to the dataset
# dataset[1] gives a dataframe, dataset$Salary gives a vector
set.seed(1234)
# regressor <- randomForest(x=dataset[1], y=dataset$Salary, ntree=10)
# regressor <- randomForest(x=dataset[1], y=dataset$Salary, ntree=100)
regressor <- randomForest(x=dataset[1], y=dataset$Salary, ntree=500)

# Predicting a new result
y_pred <- predict(regressor, data.frame(Level=6.5))

# Visualising the Random Forest Regression Model results
# Higher resolution and smoother curve - good for non-continuous regression models
x_grid <- seq(min(dataset$Level), max(dataset$Level), 0.01)
ggplot() +
  geom_point(aes(x=dataset$Level, y=dataset$Salary), color="red") +
  geom_line(aes(x=x_grid, y=predict(regressor, newdata=data.frame(Level=x_grid))),
            color="blue") +
  ggtitle("Truth or Bluff (Random Forest Regression Model)") +
  xlab("Level") +
  ylab("Salary")

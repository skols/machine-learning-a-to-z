# Decision Tree Regression - Non-continuous model
# More powerful model when have more than one dimension

# Loading libraries
library(ggplot2)
# install.packages("rpart")
library(rpart) # Decision Tree

# Importing the dataset
setwd("C:/Development/Courses/Kirill Eremenko Data Science Courses/Machine_Learning_A-Z/Part2_Regression/Section8_Decision_Tree_Regression")
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

# Fitting Decision Tree Regression Model to the dataset
regressor <- rpart(formula=Salary ~ ., data=dataset,
                   control=rpart.control(minsplit=1))

# Predicting a new result
y_pred <- predict(regressor, data.frame(Level=6.5))

# Visualising the Decision Tree Regression Model results
ggplot() +
  geom_point(aes(x=dataset$Level, y=dataset$Salary), color="red") +
  geom_line(aes(x=dataset$Level, y=predict(regressor, newdata=dataset)),
            color="blue") +
  ggtitle("Truth or Bluff (Decision Tree Regression Model)") +
  xlab("Level") +
  ylab("Salary")


# Visualising the Decision Tree Regression Model results
# Higher resolution and smoother curve - Better for Decision Tree
x_grid <- seq(min(dataset$Level), max(dataset$Level), 0.01)
ggplot() +
  geom_point(aes(x=dataset$Level, y=dataset$Salary), color="red") +
  geom_line(aes(x=x_grid, y=predict(regressor, newdata=data.frame(Level=x_grid))),
            color="blue") +
  ggtitle("Truth or Bluff (Decision Tree Regression Model)") +
  xlab("Level") +
  ylab("Salary")

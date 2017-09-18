# Polynomial Regression

# Importing the dataset
setwd("C:/Development/Courses/Kirill Eremenko Data Science Courses/Machine_Learning_A-Z/Part2_Regression/Section6_Polynomial_Regression")
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

# Fitting Linear Regression to the dataset
lin_reg <- lm(formula=Salary ~ ., data=dataset)
summary(lin_reg)

# Fitting Polynomial Regression to the dataset
dataset$Level2 <- dataset$Level^2
dataset$Level3 <- dataset$Level^3
dataset$Level4 <- dataset$Level^4
poly_reg <- lm(formula=Salary ~ ., data=dataset)
summary(poly_reg)

# Visualising the Linear Regression results
library(ggplot2)
ggplot() +
  geom_point(aes(x=dataset$Level, y=dataset$Salary), color="red") +
  geom_line(aes(x=dataset$Level, y=predict(lin_reg, newata=dataset)),
            color="blue") +
  ggtitle("Truth or Bluff (Linear Regression)") +
  xlab("Level") +
  ylab("Salary")

# Visualising the Polynomial Regression results
ggplot() +
  geom_point(aes(x=dataset$Level, y=dataset$Salary), color="red") +
  geom_line(aes(x=dataset$Level, y=predict(poly_reg, newata=dataset)),
            color="blue") +
  ggtitle("Truth or Bluff (Polynomial Regression)") +
  xlab("Level") +
  ylab("Salary")

# Predicting a new result with Linear Regression
y_pred <- predict(lin_reg, data.frame(Level=6.5))

# Predicting a new result with Polynomial Regression
y_pred <- predict(poly_reg, data.frame(Level=6.5,
                                       Level2=6.5^2,
                                       Level3=6.5^3,
                                       Level4=6.5^4))


# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
x_grid <- seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x=dataset$Level, y=dataset$Salary), color="red") +
  geom_line(aes(x=x_grid, y=predict(poly_reg,
                                    newdata=data.frame(Level=x_grid,
                                                      Level2=x_grid^2,
                                                      Level3=x_grid^3,
                                                      Level4=x_grid^4))),
            color="blue") +
  ggtitle("Truth or Bluff (Polynomial Regression)") +
  xlab("Level") +
  ylab("Salary")

# Multiple Linear Regression

# Importing the dataset
setwd("C:/Development/Courses/Kirill Eremenko Data Science Courses/Machine_Learning_A-Z/Part2_Regression/Section5_Multiple_Linear_Regression")
dataset <- read.csv("50_Startups.csv")
# dataset <- dataset[, 2:3]

# Encoding categorical data
# Transform State to a column of factors
dataset$State <- factor(dataset$State,
                        levels=c("New York", "California", "Florida"),
                        labels=c(1, 2, 3))

# Splitting the dataset into Training set and Test set
library(caTools)
set.seed(123)  # like random_state
split <- sample.split(dataset$Profit, SplitRatio=0.8)  # for training set
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

# Feature Scaling
# training_set[, 2:3] <- scale(training_set[, 2:3])  # Excluding categories since they're factors
# test_set[, 2:3] <- scale(test_set[, 2:3])

# Fitting Multiple Linear Regression to the Training set
regressor <- lm(formula=Profit ~ ., data=training_set)
# Creates the dummy variables and removes one automatically so as to not fall into the Dummy Variable trap

# To create State dummy variables; need to do before making it a factor
# dataset$New_York <- as.numeric(dataset[ dataset$State=="New York", ])
# dataset$California <- as.numeric(dataset[ dataset$State=="California", ])
# dataset$Florida <- as.numeric(dataset[ dataset$State=="Florida", ]) 
# dataset$State <- NULL

# View statistical results
summary(regressor)

# Predicting the Test set results
y_pred <- predict(regressor, newdata=test_set)

# Building the optimal model using Backward Elimination
regressor <- lm(formula=Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
                data=dataset)

summary(regressor)

# Remove State
regressor <- lm(formula=Profit ~ R.D.Spend + Administration + Marketing.Spend,
                data=dataset)

summary(regressor)

# Remove Administration
regressor <- lm(formula=Profit ~ R.D.Spend + Marketing.Spend,
                data=dataset)

summary(regressor)  # my preferred final

# Remove Marketing.Spend, but not final word because p-value is 0.06, so other considerations
regressor <- lm(formula=Profit ~ R.D.Spend,
                data=dataset)

summary(regressor)

# Kernel SVM

# Importing the dataset
setwd("C:/Development/Courses/Kirill Eremenko Data Science Courses/Machine_Learning_A-Z/Part3_Classification/Section17_Kernel_SVM")
dataset <- read.csv("Social_Network_Ads.csv")
dataset <- dataset[, 3:5]

# Splitting the dataset into Training set and Test set
library(caTools)
set.seed(123)  # like random_state
split <- sample.split(dataset$Purchased, SplitRatio=0.75)  # for training set
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

# Feature Scaling
training_set[, 1:2] <- scale(training_set[, 1:2])
test_set[, 1:2] <- scale(test_set[, 1:2])

# Fitting Kernel SVM to the Training set
library(e1071)
classifier <- svm(formula=Purchased ~ .,
                  data=training_set,
                  type="C-classification",
                  kernel="radial")  # radial basic is Gaussian

# Predicting the Test set results
y_pred <- predict(classifier, newdata=test_set[-3])  # remove last column of test set

# Making the Confusion Matrix
cm = table(test_set[, 3], y_pred)

# Visualising the Training set results - Points are truth and region is prediction
# install.packages("ElemStatLearn")
library(ElemStatLearn)
set <- training_set
X1 <- seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by=0.01)
X2 <- seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by=0.01)
grid_set <- expand.grid(X1, X2)
colnames(grid_set) <- c("Age", "EstimatedSalary")
y_grid <- predict(classifier, newdata=grid_set)
plot(set[, -3],
     main="Kernel SVM (Training Set)",
     xlab="Age", ylab="Estimated Salary",
     xlim=range(X1), ylim=range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add=TRUE)
points(grid_set, pch=19, col=ifelse(y_grid==1, "springgreen3", "tomato"))  # pch=19 for smooth region
points(set, pch=21, bg=ifelse(set[, 3]==1, "green4", "red3"))

# Visualising the Test set results - Points are truth and region is prediction
# library(ElemStatLearn)
set <- test_set
X1 <- seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by=0.01)
X2 <- seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by=0.01)
grid_set <- expand.grid(X1, X2)
colnames(grid_set) <- c("Age", "EstimatedSalary")
y_grid <- predict(classifier, newdata=grid_set)
plot(set[, -3],
     main="Kernel SVM (Test Set)",
     xlab="Age", ylab="Estimated Salary",
     xlim=range(X1), ylim=range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add=TRUE)
points(grid_set, pch=19, col=ifelse(y_grid==1, "springgreen3", "tomato"))
points(set, pch=21, bg=ifelse(set[, 3]==1, "green4", "red3"))

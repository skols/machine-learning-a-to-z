# Artificial Neural Network

# Importing the dataset
setwd("C:/Development/Courses/Kirill Eremenko Data Science Courses/Machine_Learning_A-Z/Part8_Deep_Learning/Section31_Artificial_Neural_Networks")
dataset <- read.csv("Churn_Modelling.csv")
dataset <- dataset[, 4:14]

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

# Feature Scaling
training_set[-11] <- scale(training_set[-11])
test_set[-11] <- scale(test_set[-11])

# Fitting ANN to the Training set
# install.packages("h2o")
library(h2o)
h2o.init(nthreads=-1)
# hidden=c(6, 6) means 6 neurons in hidden layer 1, 6 in hidden layer 2. Continue for each hidden layer.
classifier <- h2o.deeplearning(y="Exited", training_frame=as.h2o(training_set),
                               activation="Rectifier", hidden=c(6, 6),
                               epochs=100, train_samples_per_iteration=-2)

# Predicting the Test set results
prob_pred <- h2o.predict(classifier, newdata=as.h2o(test_set[-11]))  # remove last column of test set
y_pred <- (prob_pred > 0.5)
y_pred <- as.vector(y_pred)

# Making the Confusion Matrix
cm = table(test_set[, 11], y_pred)
(1522 + 206)/2000

h2o.shutdown()

# Natural Language Processing

# Importing the dataset
setwd("C:/Development/Courses/Kirill Eremenko Data Science Courses/Machine_Learning_A-Z/Part7_Natural_Language_Processing/Section29_Natural_Language_Processing")
dataset_original <- read.delim("Restaurant_Reviews.tsv", quote="", stringsAsFactors=FALSE)

# Cleaning the text
library(tm)
library(SnowballC)  # stopwords function
corpus <- VCorpus(VectorSource(dataset$Review))
# as.character(corpus[[1]]) gives first review

# Converting all reviews to lowercase
corpus <- tm_map(corpus, content_transformer(tolower))

# Remove numbers
corpus <- tm_map(corpus, removeNumbers)

# Remove puncuation
corpus <- tm_map(corpus, removePunctuation)

# Remove non-significant words
corpus <- tm_map(corpus, removeWords, stopwords())

# Stemming
corpus <- tm_map(corpus, stemDocument)

# Remove any extra spaces
corpus <- tm_map(corpus, stripWhitespace)

# Creating the Bag of Words model
dtm <- DocumentTermMatrix(corpus)

# Filter words to keep only most frequent
dtm <- removeSparseTerms(dtm, 0.999)  # keep 99.9%; removes close to 1000 words

# Random Forest Classification
dataset <- as.data.frame(as.matrix(dtm))
dataset$Liked <- dataset_original$Liked

# Encoding the target feature as factor
dataset$Liked <- factor(dataset$Liked, levels=c(0, 1))

# Splitting the dataset into Training set and Test set
library(caTools)
set.seed(123)  # like random_state
split <- sample.split(dataset$Liked, SplitRatio=0.8)  # for training set
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

# Fitting Random Forest Classification to the Training set
library(randomForest)
classifier <- randomForest(x=training_set[-length(dataset)],
                           y=training_set$Liked,
                           ntree=10)

# Predicting the Test set results
y_pred <- predict(classifier, newdata=test_set[-length(dataset)])  # remove last column of test set

cm_stats <- function(y_test, y_pred, class_type) {
  cm <- table(test_set[, length(dataset)], y_pred)
  accuracy <- (cm[[1]] + cm[[4]])/nrow(y_test)
  precision <- cm[[4]]/(cm[[4]] + cm[[3]])
  recall <- cm[[4]]/(cm[[4]] + cm[[2]])
  f1_score <- 2 * precision * recall / (precision + recall)
  print(class_type)
  print(cm)
  paste("accuracy: ", accuracy, "precision: ", precision, "recall: ", recall, "f1_score: ", f1_score)
  # paste("precision: ", precision)
  # paste("recall: ", recall)
  # paste("f1_score: ", f1_score)
}

cm_stats(test_set, y_pred, "Random Forest")


# Fitting Decision Tree to the Training set
library(rpart)
classifier <- rpart(formula=Liked ~ .,
                    data=training_set)

# Predicting the Test set results
y_pred <- predict(classifier, newdata=test_set[-length(dataset)], type="class")  # remove last column of test set
# Use type="class" so don't get a matrix of results

cm_stats(test_set, y_pred, "Decision Tree")

# Fitting Logistic Regression to the Training set
classifier <- glm(formula=Liked ~ .,
                  family=binomial,
                  data=training_set)

# Predicting the Test set results
prob_pred <- predict(classifier, type="response", newdata=test_set[-length(dataset)])  # remove last column of test set
y_pred <- ifelse(prob_pred > 0.5, 1, 0)

cm_stats(test_set, y_pred, "Logistic Regression")

# Fitting K-NN to the Training set and Predicting the Test set results
library(class)
# Only want first two columns of training_set and test_set
y_pred <- knn(train=training_set[, -length(dataset)],
              test=test_set[, -length(dataset)],
              cl=training_set[, length(dataset)],
              k=5)

cm_stats(test_set, y_pred, "K-NN")

# Fitting SVM to the Training set
library(e1071)
classifier <- svm(formula=Liked ~ .,
                  data=training_set,
                  type="C-classification",
                  kernel="linear")

# Predicting the Test set results
y_pred <- predict(classifier, newdata=test_set[-length(dataset)])  # remove last column of test set

cm_stats(test_set, y_pred, "SVM")

# Fitting Kernel SVM to the Training set
library(e1071)
classifier <- svm(formula=Liked ~ .,
                  data=training_set,
                  type="C-classification",
                  kernel="radial")  # radial basic is Gaussian

# Predicting the Test set results
y_pred <- predict(classifier, newdata=test_set[-length(dataset)])  # remove last column of test set

cm_stats(test_set, y_pred, "Kernel SVM")

# Fitting Naive Bayes to the Training set
library(e1071)
classifier <- naiveBayes(x=training_set[-length(dataset)],
                         y=training_set$Liked)

# Predicting the Test set results
y_pred <- predict(classifier, newdata=test_set[-length(dataset)])  # remove last column of test set

cm_stats(test_set, y_pred, "Naive Bayes")

# Fitting Maximum Entropy to the Training set
# install.packages("maxent")
library(maxent)
classifier <- maxent(feature_matrix=training_set[-length(dataset)],
                         code_vector=training_set$Liked)

# Predicting the Test set results
y_pred <- predict(classifier, test_set[-length(dataset)])  # remove last column of test set

cm <- table(test_set[, length(dataset)], as.factor(y_pred[, 1]))
accuracy <- (cm[[1]] + cm[[4]])/nrow(test_set)
precision <- cm[[4]]/(cm[[4]] + cm[[3]])
recall <- cm[[4]]/(cm[[4]] + cm[[2]])
f1_score <- 2 * precision * recall / (precision + recall)
print("Maximum Entropy")
print(cm)
paste("accuracy: ", accuracy, "precision: ", precision, "recall: ", recall, "f1_score: ", f1_score)

# Fitting C5.0 to the Training set
library(C50)
classifier <- C5.0(x=training_set[-length(dataset)],
                         y=training_set$Liked)

# Predicting the Test set results
y_pred <- predict(classifier, newdata=test_set[-length(dataset)])  # remove last column of test set

cm_stats(test_set, y_pred, "C5.0")

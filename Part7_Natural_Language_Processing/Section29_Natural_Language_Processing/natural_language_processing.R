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

# Making the Confusion Matrix
cm = table(test_set[, length(dataset)], y_pred)

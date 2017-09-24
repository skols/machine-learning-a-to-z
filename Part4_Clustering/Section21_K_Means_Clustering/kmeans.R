# K-Means Clustering

# Importing the mall dataset
setwd("C:/Development/Courses/Kirill Eremenko Data Science Courses/Machine_Learning_A-Z/Part4_Clustering/Section21_K_Means_Clustering")
dataset <- read.csv("Mall_Customers.csv")
X <- dataset[4:5]

# Using the elbow method to find the optimal number of clusters
set.seed(6)
wcss <- vector()  # initialize an empty vector
for (i in 1:10) wcss[i] <- sum(kmeans(X, i)$withinss)

plot(1:10, wcss, type="b", main=paste("Clusters of clients"), xlab="Number of clusters",
     ylab="WCSS")
# 5 clusters

# Applying k-means to the mall dataset
set.seed(29)
kmeans <- kmeans(X, 5, iter.max=300, nstart=10)

# Visualising the clusters
library(cluster)
clusplot(X, kmeans$cluster, lines=0, shade=TRUE, color=TRUE, labels=2,
         plotchar=FALSE, span=TRUE, main=paste("Clusters of clients"),
         ylab="Annual Income", xlab="Spending Score")

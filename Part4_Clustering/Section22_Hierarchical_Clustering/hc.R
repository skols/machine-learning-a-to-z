# Hierarchical Clustering

# Importing the mall dataset
setwd("C:/Development/Courses/Kirill Eremenko Data Science Courses/Machine_Learning_A-Z/Part4_Clustering/Section22_Hierarchical_Clustering")
dataset <- read.csv("Mall_Customers.csv")
X <- dataset[4:5]

# Using the dendrogram to find the optimal number of clusters
dendogram <- hclust(dist(X, method="euclidean"), method="ward.D")
plot(dendogram, main=paste("Dendrogram"), xlab="Customers",
     ylab="Euclidean distances")
# 5 clusters

# Fitting hierarchical clustering to the mall dataset
hc <- hclust(dist(X, method="euclidean"), method="ward.D")
y_hc <- cutree(hc, 5)

# Visualising the clusters
library(cluster)
clusplot(X, y_hc, lines=0, shade=TRUE, color=TRUE, labels=2,
         plotchar=FALSE, span=TRUE, main=paste("Clusters of clients"),
         ylab="Annual Income", xlab="Spending Score")

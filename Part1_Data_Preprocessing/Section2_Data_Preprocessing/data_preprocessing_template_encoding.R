# Data Preprocessing

# Importing the dataset
setwd("C:/Development/Courses/Kirill Eremenko Data Science Courses/Machine_Learning_A-Z/Part1_Data_Preprocessing/Section2_Data_Preprocessing")
dataset = read.csv("Data.csv")

# Encoding categorical data
# Transform country to a column of factors
dataset$Country = factor(dataset$Country,
                         levels=c("France", "Spain", "Germany"),
                         labels=c(1, 2, 3))

# Transform purchased to a factor
dataset$Purchased = factor(dataset$Purchased,
                         levels=c("No", "Yes"),
                         labels=c(0, 1))

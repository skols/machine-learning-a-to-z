# Data Preprocessing

# Importing the dataset
setwd("C:/Development/Courses/Kirill Eremenko Data Science Courses/Machine_Learning_A-Z/Part1_Data_Preprocessing/Section2_Data_Preprocessing")
dataset <- read.csv("Data.csv")

# Take care of missing data
dataset$Age <- ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN=function(x) mean(x, na.rm=TRUE)),
                     dataset$Age)

dataset$Salary <- ifelse(is.na(dataset$Salary),
                     ave(dataset$Salary, FUN=function(x) mean(x, na.rm=TRUE)),
                     dataset$Salary)

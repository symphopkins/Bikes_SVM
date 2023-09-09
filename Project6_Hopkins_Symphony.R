#############################################
#                                           #
# Author:     Symphony Hopkins              #
# Date:       04/21/2023                    #
# Subject:    Project 6                     #
# Class:      DSCI 512                      #
# Section:    01W                           #         
# Instructor: Juan David Munoz              #
# File Name:  Project6_Hopkins_Symphony.R   #
#                                           #
#############################################

#1. Load the dataset bike.csv into memory. Convert 
#   holiday to a factor using factor() function. Then 
#   split the data into training set containing 2/3 of 
#   the original data (test set containing remaining 1/3 
#   of the original data).
#   Answer: See code.

#importing library
bike <- read.csv("~/Documents/Maryville_University/DSCI_512/Bike.csv")
View(bike)

#converting holiday to factor
bike$holiday <- as.factor(bike$holiday)

#setting seed for reproducibility
set.seed(1)

#performing train-test split
train = sample(1:nrow(bike), nrow(bike) * 2/3)

#2. Build a support vector machine model.
#   Answer: See following questions.

#importing library
library(e1071)

#2a.The response is holiday and the predictors are: 
#   season, workingday, casual, and registered. Please 
#   use svm() function with radial kernel and gamma=10 
#   and cost = 100.
#   Answer: See code.

#creating support vector machine
bike_svm <- svm(holiday ~ season + workingday + casual + registered,
                data=bike[train, ], kernel='radial', gamma=100, cost=100)
summary(bike_svm)

#2b.Perform a grid search to find the best model 
#   with potential cost: 1, 10, 50, 100 and potential 
#   gamma: 1, 3, and 5 and using radial kernel and training 
#   dataset.
#   Answer: See code.

#performing grid search
bike_grid_search <- tune(svm, holiday ~ season + workingday + casual + registered,
                     data=bike[train, ], kernel='radial', ranges=list(cost=c(1, 10, 50, 100),
                      gamma=c(1, 3, 5)))

#2c.Print out the model results. What’s the best model parameters?
#   Answer: The best model occurs when the parameters are as the
#   following: cost=100 and gamma=1.

#printing model results
print(bike_grid_search)

#2d.Forecast holiday using the test dataset and the best 
#   model found in c).
#   Answer: See code.

#forecasting holiday
bike_pred <- predict(bike_grid_search$best.model, newdata=bike[-train,])

#2e.Get the true observations of holiday in the test dataset.
#   Answer: See code.

#finding true observations
bike_true_obs <- bike[-train, 'holiday']

#2f.Compute the test error by constructing the confusion matrix. 
#   Is it a good model?
#   Answer: This is a good model, with 97% of the predictions being 
#   correct (true positive and true negative).

#creating confusion matrix
bike_cm <- table(bike_true_obs, bike_pred)
print(bike_cm)

#calculating percentage of correct predictions
total <- 3529 + 3 + 89 + 8
total_correct <- (3529 + 8)/ total
print(total_correct)

#End Assignment





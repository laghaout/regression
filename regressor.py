# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 10:12:10 2015

@author: Amine Laghaout

Template for a regression script
"""

# %% Input parameters

trainFile = 'trainData.txt'    # Training data
testFile = 'testData.txt'      # Test data
delim = '\t'                        # Feature delimiter

# Number of entries to be loaded. Set None to load all entries. For 
# a quicker run, load only a small number of rows.
maxRows = 1000 

# %% Load the data

import numpy as np
from pandas import read_csv

# Load the training data
trainData = np.array(read_csv(trainFile, nrows = maxRows, delimiter = delim))

# Load the test data
testFeatures = np.array(read_csv(testFile, delimiter = delim))

# Determine the targets, features, the number of data points M and the 
# number of features N from the training data
targets = np.array([elem for elem in trainData[:, 0]])
features = trainData[:, 1:]
(M, N) = np.shape(features)

# %% Wrangle the data so that it can be consumed for regression

from regressorModule import ImputeMissing, HotEncode, SplitData

# Determine the indices of the columns containing categorical features.
# We assume that the test set has the same feature data types as the 
# trainig set.
categoricalIndices = [list(features[0]).index(elem) \
    for elem in features[0] if isinstance(elem, str)]

# Replace the missing categorical features by the empty string
for strCol in categoricalIndices:
    features[:, strCol] = [elem if isinstance(elem, str) else '' for elem in features[:, strCol]]
    testFeatures[:, strCol] = [elem if isinstance(elem, str) else '' for elem in testFeatures[:, strCol]]

# Encode the categorical features into numbers
for strCol in categoricalIndices:
    features[:, strCol] = HotEncode(features[:, strCol])
    testFeatures[:, strCol] = HotEncode(testFeatures[:, strCol])

# Replace any  missing feature by the mean of its specified values 
features = ImputeMissing(features)
testFeatures = ImputeMissing(testFeatures)

# %% Split the data into training and validation sets

validationFraction = 1/3 # Fraction of the data used for validation

trainFeatures, validFeatures, trainTargets, validTargets, maxRow\
= SplitData(features, targets, validationFraction)

# %% Train

# Alternative estimators I played around with.
#from sklearn import linear_model
#regr = linear_model.Lasso(alpha = 200)
#regr = linear_model.Ridge(alpha = 200)
#regr = linear_model.RidgeCV(alphas=[0.1, 1, 10, 100, 200, 300, 400]) # Regularization term
#regr = linear_model.BayesianRidge()

from sklearn import svm

# Train the data from the training subset (i.e., minus the validation
# subset). We will use this to compute figures of merit such as the 
# root mean squared and the prediction score.
regr = svm.SVR(C = 30)
regr.fit(trainFeatures, trainTargets)

# Train the data set on the entire training set (i.e., without 
# splitting it into training proper and validation). We will use this
# to predict the data from the test set.
regrOut = svm.SVR(C = 30)
regrOut.fit(features, targets)

# %% Performance report

# Predict from the validation features
predictedValidate = regr.predict(validFeatures)

# Predict from the training features only
predictedTrain = regr.predict(trainFeatures)

# Predict from the whole training set (training+validation features)
predicted = regr.predict(features)

# Compute the root mean squared (RMS) for each case above

from sklearn.metrics import mean_squared_error
from math import sqrt

RMSvalid = sqrt(mean_squared_error(validTargets, predictedValidate))
RMStrain = sqrt(mean_squared_error(trainTargets, predictedTrain))
RMS = sqrt(mean_squared_error(targets, predicted))

# Print to standard IO the RMS and prediction scores.

print('RMS on validation set = ', RMSvalid)
print('RMS on training set = ', RMStrain)
print('RMS on whole set = ', RMS)

print('Score on validation set = ', regr.score(validFeatures, validTargets))
print('Score on training set = ', regr.score(trainFeatures, trainTargets))
print('Score on whole set = ', regr.score(features, targets))

# %% This is used solely to have a visual comparison of the targets 
# from the colored variable explorer of the Spyder IDE

from regressorModule import CompareVectors

CompareValid = CompareVectors(predictedValidate, validTargets)
CompareTrain = CompareVectors(predictedTrain, trainTargets)
Compare = CompareVectors(predicted, targets)

# %% Predict the targets from the test data and save them to file

testTargets = regrOut.predict(testFeatures)
np.savetxt('testTargets.txt', testTargets, newline = '\n')


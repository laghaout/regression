# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 10:15:25 2015

@author: Amine Laghaout
"""

def ImputeMissing(feat):

    '''
    This function replaces the missing feat by the means of the 
    columns.
    http://scikit-learn.org/stable/modules/preprocessing.html#imputation-of-missing-values

    INPUT: The feature data with possible missing entries
    
    OUTPUT: The feature data with the missing entries filled by the
    means of each concerned feature
    '''

    from sklearn.preprocessing import Imputer

    imp = Imputer(missing_values = 'NaN', strategy = 'mean', axis=0)
    imp.fit(feat)
    Imputer(axis = 0, copy = True, missing_values = 'NaN', 
            strategy = 'mean', verbose = 0)
    feat = imp.transform(feat)    
    
    return feat

def HotEncode(column):

    '''
    This function encodes the categorical features (i.e., strings) into
    scalars. OneHotEncoder should be added to this implementation.
    
    INPUT: The feature column of string categorical data
    
    OUTPUT: The feature column of scalar-encoded categorical data
    '''

    from sklearn import preprocessing
    
    le = preprocessing.LabelEncoder()
    le.fit(column)
    column = le.transform(column)
    
    # Instead of simply mapping the categorical features onto a scalar,
    # one shoud instead encode them into an array of integers. This is 
    # yet to be implemented.
    '''
    enc = preprocessing.OneHotEncoder()
    enc.fit(column)
    column = enc.transform(column)
    '''
    
    return column
    
def CompareVectors(predicted, actual):

    '''
    Assembles the predicted and targets vectors side by side so that
    they can be inspected easily from the variable explorer of Spyder.
    
    INPUT: Two vectors
    
    OUTPUT: The two vectors assembled side by side into a matrix
    '''

    from numpy import zeros

    ComparisonMartix = zeros([len(predicted), 2])
    ComparisonMartix[:,0] = predicted
    ComparisonMartix[:,1] = actual
    
    return ComparisonMartix
    
def SplitData(feat, targets, validationFraction):

    ''' 
    Splits the data into training and validation sets 
    
    INPUT: The feature and target data as well as the fraction that
    should be assigned to validation
    
    OUTPUT: The feature and target data for training proper and for 
    validation as well as the number of rows assigned to training 
    proper 
    '''

    from numpy import shape

    # Hard coded version: Without randomization.
    '''
    maxRow = np.floor((1-validationFraction)*M)
    
    featuresTrain = feat[0:maxRow, :]
    targetsTrain = targets[0:maxRow]
    
    featuresValidate = feat[maxRow:, :]
    targetsValidate = targets[maxRow:]
    '''
    
    from sklearn.cross_validation import train_test_split
    
    featuresTrain, featuresValidate, targetsTrain, targetsValidate = \
        train_test_split(feat, targets, test_size = validationFraction, 
                         random_state = 0)
    maxRow = shape(featuresTrain)[0]
    
    return featuresTrain, featuresValidate, targetsTrain, \
        targetsValidate, maxRow
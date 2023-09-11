# Predicting Elemental Weight Percents from Hyperspectral Core Image Data using XRF and Machine Learning
# 
# 
# By Zach Toews modified
# 
# Department of Geoscience
# 
# The University of Calgary
# 
# **********************************************************************************************************************
# 
# **Approach:**
# 1. XRF and hyperspectral data was cleaned in Excel. 
# 2. Only the hyperspectral data that matches with an XRF data point are included in the training set.
# 3. The training data is used to build and tune a model.
# 4. An XRF prediction is then made on every hyperspectal data point using the model.
# 
# **********************************************************************************************************************


# IMPORT AND DATA LOADING
# import packages
import os                                                   # Operating system
import pandas as pd                                         # Dataframes
import numpy as np                                          # Arrays
from sklearn.linear_model import LinearRegression           # linear regression
from sklearn.metrics import r2_score                        # R2 score
from sklearn.model_selection import train_test_split        # train and test split
from sklearn.ensemble import RandomForestRegressor          # Random Forest
from sklearn.ensemble import GradientBoostingRegressor      # Gradient Boosting
import warnings

# ignore warnings
warnings.filterwarnings("ignore")


def linear_regression(X, Y):
    '''
    Function that accepts a datatframe of X data and a dataframe of Y data and uses linear regression 
    to fit 80% of the X and Y data and 20% is used for testing. This  returns a dataframe of the True and 
    Predicted values for the test data and a list of r squared values for each Y column.
    
    Args:
        X (dataframe): dataframe of X data where the first column is depth and the rest are used for prediction
        Y (dataframe): dataframe of Y data where the first column is depth and the rest are used for prediction
        
    Returns:
        test_data (dataframe): dataframe of True and Predicted values for the test data
        r2_values (list): list of r squared values for each Y column
    '''
    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X.iloc[:, 1:], Y.iloc[:, 1:], test_size=0.2, random_state=0)

    # Fit a linear regression model to the training data
    model = LinearRegression()
    model.fit(X_train, Y_train)

    # Predict the Y values for the test data
    Y_pred = model.predict(X_test)

    # Create a dataframe of the true and predicted values for the test data
    test_data = pd.DataFrame(np.concatenate((Y_test.values, Y_pred), axis=1), columns=Y.columns)

    # renmae the columns to true and predicted
    test_data.columns = ["true", "predicted"]

    # Calculate the r squared values for each Y column
    r2_values = [model.score(X_test, Y_test[col]) for col in Y.columns[1:]]

    return test_data, r2_values


def random_forest_regression(X, Y, bootstrap=True, max_depth=80, max_features='sqrt', min_samples_leaf=1, min_samples_split=10, n_estimators=500):
    '''
    Function that accepts a datatframe of X data and a dataframe of Y data and uses random forest regression 
    to fit 80% of the X and Y data and 20% is used for testing. This  returns a dataframe of the True and 
    Predicted values for the test data and a list of r squared values for each Y column.
    
    Args:
        X (dataframe): dataframe of X data where the first column is depth and the rest are used for prediction
        Y (dataframe): dataframe of Y data where the first column is depth and the rest are used for prediction
        
    Returns:
        test_data (dataframe): dataframe of True and Predicted values for the test data
        r2_values (list): list of r squared values for each Y column
    '''
    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X.iloc[:, 1:], Y.iloc[:, 1:], test_size=0.2, random_state=0)

    # Fit a random forest regression model to the training data
    model = RandomForestRegressor(bootstrap=True, max_depth=max_depth, max_features=max_features, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, n_estimators=n_estimators)
    model.fit(X_train, Y_train)

    # Predict the Y values for the test data
    Y_pred = model.predict(X_test)

    # Create a dataframe of the true and predicted values for the test data
    test_data = pd.DataFrame(np.concatenate((Y_test.values, Y_pred.reshape(-1, 1)), axis=1), columns=Y.columns)

    # renmae the columns to true and predicted
    test_data.columns = ["true", "predicted"]

    # Calculate the r squared values for each Y column
    r2_values = [model.score(X_test, Y_test[col]) for col in Y.columns[1:]]

    return test_data, r2_values


def gradient_boosting_regression(X, Y, learning_rate=0.1, max_depth=80, max_features='sqrt', min_samples_leaf=1, min_samples_split=10, n_estimators=500):
    '''
    Function that accepts a datatframe of X data and a dataframe of Y data and uses gradient boosting regression
    to fit 80% of the X and Y data and 20% is used for testing. This  returns a dataframe of the True and 
    Predicted values for the test data and a list of r squared values for each Y column.
    
    Args:
        X (dataframe): dataframe of X data where the first column is depth and the rest are used for prediction
        Y (dataframe): dataframe of Y data where the first column is depth and the rest are used for prediction
        
    Returns:
        test_data (dataframe): dataframe of True and Predicted values for the test data
        r2_values (list): list of r squared values for each Y column
    '''
    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X.iloc[:, 1:], Y.iloc[:, 1:], test_size=0.2, random_state=0)

    # Fit a gradient boosting regression model to the training data
    model = GradientBoostingRegressor(learning_rate=0.1, max_depth=max_depth, max_features=max_features, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, n_estimators=n_estimators)
    model.fit(X_train, Y_train)

    # Predict the Y values for the test data
    Y_pred = model.predict(X_test)

    # Create a dataframe of the true and predicted values for the test data
    test_data = pd.DataFrame(np.concatenate((Y_test.values, Y_pred.reshape(-1, 1)), axis=1), columns=Y.columns)

    # renmae the columns to true and predicted
    test_data.columns = ["true", "predicted"]

    # Calculate the r squared values for each Y column
    r2_values = [model.score(X_test, Y_test[col]) for col in Y.columns[1:]]

    return test_data, r2_values

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
from sklearn.linear_model import LinearRegression           # linear regression
from sklearn.metrics import r2_score                        # R2 score
from sklearn.model_selection import train_test_split        # train and test split
from sklearn.ensemble import RandomForestRegressor          # Random Forest
import warnings

# ignore warnings
warnings.filterwarnings("ignore")


def create_XY_df(df):
# Create X and Y dataframes
    X = df
    Y = df
    for col in X.columns:
        if col[:4] != 'LWIR' and col[:4] != 'SWIR':
            X = X.drop(columns=col)

    for col in Y.columns:
        if col[:4] == 'LWIR' or col[:4] == 'SWIR':
            Y = Y.drop(columns=col)
    return X, Y

def liner_reg_training(df):
    # LINEAR REGRESSION TRAINING
    X, Y = create_XY_df(df)
    elements_r2 = []
    Y_pred = pd.DataFrame()
    n = 0
    lin_reg = LinearRegression()
    for elm in Y.columns:
        lin_reg.fit(X, Y[elm])
        elements_r2.append([])
        elements_r2[n].append(elm)
        Y_pred[elm] = lin_reg.predict(X)
        elements_r2[n].append(r2_score(Y[elm], Y_pred[elm]))
        n = n + 1

    elements_r2 = sorted(elements_r2, key=lambda x: x[1], reverse=True)

    dfExport = pd.DataFrame()
    for elm in Y.columns:
        dfExport[elm] = Y[elm]
        dfExport[elm + '_pred'] = Y_pred[elm]

    for col in X.columns:
        if col[:4] != 'LWIR' and col[:4] != 'SWIR':
            X = X.drop(columns=col)

    Y_pred_HSI = pd.DataFrame()
    for elm in Y.columns:
        #lin_reg = LinearRegression()
        lin_reg.fit(X, Y[elm])
        Y_pred_HSI[elm] = lin_reg.predict(X)

    dfExportPred = pd.DataFrame()
    for elm in Y.columns:
        dfExportPred[elm + '_pred'] = Y_pred_HSI[elm]

    return dfExport, elements_r2


def my_train_test_split(df):
    # TRAIN TEST SPLIT
    X, Y = create_XY_df(df)
    X = df
    for col in X.columns:
        if col[:4] != 'LWIR' and col[:4] != 'SWIR' and col != 'Depth':
            X = X.drop(columns=col)
    x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)
    x_train.sort_values(by=['Depth'], inplace=True)
    y_train.sort_values(by=['Depth'], inplace=True)
    x_test.sort_values(by=['Depth'], inplace=True)
    y_test.sort_values(by=['Depth'], inplace=True)
    x_test = x_test.drop(columns='Depth')
    x_train = x_train.drop(columns='Depth')

    return x_train, y_train, x_test, y_test


def ran_forest_pred(df):
    # RANDOM FOREST TEST PREDICTION
    X, Y = create_XY_df(df)
    Y_pred = pd.DataFrame()
    x_train, x_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)
    rfr = RandomForestRegressor(bootstrap=True, max_depth=80, max_features='sqrt', min_samples_leaf=1, min_samples_split=10, n_estimators=500).fit(x_train, Y_train)
    Y_pred = rfr.predict(x_test)
    elements_r2 = r2_score(Y_test, Y_pred)

    dfExport = pd.DataFrame()
    dfExport['mineral_pred'] = Y_pred
    dfExport['mineral'] = Y_test.reset_index().drop(columns='index')
    
    return dfExport, elements_r2


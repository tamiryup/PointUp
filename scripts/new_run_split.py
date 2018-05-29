import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
import datetime

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
import xgboost as xgb

from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

data = pd.read_csv("binned_clean.csv")
validation_X = []
validation_Y = []
test_X = []
test_Y = []

def run_linear_reg(X, Y):
    print("LINEAR REGRESSION: ")
    linreg = LinearRegression()
    linreg.fit(X, Y["SALE PRICE"])
    pred_y = linreg.predict(validation_X)
    measure_acc(pred_y, validation_Y)
    return linreg


def run_lasso_reg(X, Y):
    print("LASSO REGRESSION: ")
    lasso = Lasso(alpha=1)
    lasso.fit(X,Y["SALE PRICE"])
    pred_y = lasso.predict(validation_X)
    measure_acc(pred_y, validation_Y)
    return lasso

def run_ridge_reg(X, Y):
    print("RIDGE REGRESSION: ")
    ridge = Ridge(alpha=0.1)
    ridge.fit(X, Y["SALE PRICE"])
    pred_y = ridge.predict(validation_X)
    measure_acc(pred_y, validation_Y)
    return ridge

def run_random_forest(X, Y):
    print("RANDOM FOREST REGRESSION:")
    rand_forest = RandomForestRegressor()
    rand_forest.fit(X, Y["SALE PRICE"])
    pred_y = rand_forest.predict(validation_X)
    print("R^2: ", rand_forest.score(validation_X, validation_Y))
    measure_acc(pred_y, validation_Y)
    # importance = pd.DataFrame(list(zip(X.columns, np.transpose(rand_forest.feature_importances_))) \
    #                           ).sort_values(1, ascending=False)
    # print(importance)
    return rand_forest

def col_onehot_encoding(col):
    global data
    prefix = col.lower().replace(" ","_")
    data = pd.concat([data, pd.get_dummies(data[col], prefix=prefix)], axis=1)
    data.drop([col], axis=1, inplace=True)

def measure_acc(y_pred, y):
    #print(y)
    #print(y_pred)
    s = 0
    percentage = 0.1
    for i in range(y.shape[0]):
        pred = y_pred[i]
        real = y["SALE PRICE"].iloc[i]
        if float(pred)/real <1+percentage and float(pred)/real >1-percentage:
            s+=1
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mean_abs_err = mean_absolute_error(y, y_pred)
    print("number of samples in validatoin set: ", y.shape[0])
    print("number of predictions within percentage: ", s)
    print("ratio of predictoin within percentage: ", float(s)/y.shape[0])
    print("Root Mean Squared Error: ", rmse)
    print("Mean Absolute Error: ", mean_abs_err)

def main():
    global validation_X, validation_Y, test_X, test_Y
    print(data.dtypes)
    cols = ["BOROUGH", "NEIGHBORHOOD", "BUILDING CLASS CATEGORY", "TAX CLASS AT PRESENT", "BUILDING CLASS AT PRESENT", \
            "BUILDING CLASS AT TIME OF SALE", "YEAR TYPE", "TAX CLASS AT TIME OF SALE"]

    for col in cols:
        col_onehot_encoding(col)
    print(data.dtypes)

    data["SALE DATE"] = data["SALE DATE"].apply(lambda date: date.split("-")[0]).astype(int)  # this is temporary

    train_data, test_data = train_test_split(data, test_size=0.3)
    test_data, validation_data = train_test_split(test_data, test_size=0.5)
    print("data size:", data.shape)
    print("test size: ", test_data.shape)
    print("train size: ", train_data.shape)
    print("validation size: ", validation_data.shape)

    train_cols = list(train_data.columns.values)
    train_cols.remove("SALE PRICE")
    train_cols.remove("Unnamed: 0")
    train_cols.remove("ADDRESS")  # temporery
    train_cols.remove("APARTMENT NUMBER")  # temperary

    X = train_data.loc[:, train_cols]
    Y = train_data.loc[:, ["SALE PRICE"]]

    validation_X = validation_data.loc[:, train_cols]
    validation_Y = validation_data.loc[:, ["SALE PRICE"]].astype(int)

    test_X = test_data.loc[:, train_cols]
    test_Y = test_data.loc[:, ["SALE PRICE"]].astype(int)

    run_ridge_reg(X, Y)


main()
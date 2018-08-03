# Key parts of the code needed for daily work.
# 1st step - clean the dataset (rarely redone, thus in a separate file), 
# then we run the common models to evaluate progeress 
# Each function runs a different model

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
from sklearn.svm import SVR
from sklearn.feature_selection import RFE
from sklearn import neighbors
from sklearn.linear_model import ElasticNet
import xgboost as xgb
from xgboost import XGBRegressor

from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data = pd.read_csv("nyc_sales_2.0.csv")

validation_X = []
validation_Y = []
test_X = []
test_Y = []


"""
converts a pandas column of categorical variables,
into numerical numbers for each category
"""
def category_to_numbers(col):
    global data
    data[col] = data[col].str.strip()
    data[col] = data[col].astype("category")
    uniques = data[col].unique()
    dic = {uniques[k]: k for k in range(len(uniques))}
    data = data.replace({col:dic})



"""
runs linear regression
"""
def run_linear_reg(X, Y):
    print("LINEAR REGRESSION: ")
    linreg = LinearRegression()
    linreg.fit(X, Y["SALE PRICE"])
    pred_y = linreg.predict(validation_X)
    measure_acc(pred_y, validation_Y)
    return linreg


"""
runs lasso regression
"""
def run_lasso_reg(X, Y):
    print("LASSO REGRESSION: ")
    lasso = Lasso(alpha=1)
    lasso.fit(X,Y["SALE PRICE"])
    pred_y = lasso.predict(validation_X)
    measure_acc(pred_y, validation_Y)
    return lasso


"""
runs ridge regression
"""
def run_ridge_reg(X, Y):
    print("RIDGE REGRESSION: ")
    ridge = Ridge(alpha=0.1)
    ridge.fit(X, Y["SALE PRICE"])
    pred_y = ridge.predict(validation_X)
    measure_acc(pred_y, validation_Y)
    return ridge



"""
runs random forest regressor
"""
def run_random_forest(X, Y):
    print("RANDOM FOREST REGRESSION:")
    rand_forest = RandomForestRegressor()
    rand_forest.fit(X, Y["SALE PRICE"])
    pred_y = rand_forest.predict(validation_X)
    print("R^2: ", rand_forest.score(validation_X, validation_Y))
    measure_acc(pred_y, validation_Y)
    importance = pd.DataFrame(list(zip(X.columns, np.transpose(rand_forest.feature_importances_))) \
                              ).sort_values(1, ascending=False)
    print(importance)
    return rand_forest


"""
runs random forest rfe
"""
def run_random_forest_rfe(X, Y):
    print("RANDOM FOREST REGRESSION:")
    rand_forest = RandomForestRegressor()
    rfe=RFE(estimator=rand_forest)
    rfe.fit(X, Y["SALE PRICE"])
    pred_y = rfe.predict(validation_X)
    #print("R^2: ", rfe.score(validation_X, validation_Y))
    print("number of features used: ", rfe.n_features_)
    measure_acc(pred_y, validation_Y)
    # importance = pd.DataFrame(list(zip(X.columns, np.transpose(rand_forest.feature_importances_))) \
    #                           ).sort_values(1, ascending=False)
    # print(importance)
    return rfe



"""
runs xgboost regressor
"""
def run_xgboost(X, Y):
    xgb_model = XGBRegressor()
    xgb_model.fit(X, Y)
    pred_y = xgb_model.predict(validation_X)
    measure_acc(pred_y, validation_Y)
    return xgb_model

"""
runs knn
"""
def run_knn(X, Y):
    print("NEAREST NEIGHBORS REGRESSION (KNN):")
    n_neighbors=10
    knn = neighbors.KNeighborsRegressor(n_neighbors)
    knn.fit(X, Y)
    pred_y = knn.predict(validation_X)
    measure_acc(pred_y, validation_Y)
    return knn

"""
defines the baseline model which is a sequential neural net using Keras
"""
def baseline_model():
    model = Sequential()
    model.add(Dense(370, input_dim=370, kernel_initializer='normal', activation='relu'))
    model.add(Dense(200, kernel_initializer='normal', activation='relu'))
    model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

"""
runs the keras regressor using the baseline neural net defined above
"""
def run_nn(X,Y):
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)
    estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=2)
    estimator.fit(X, Y)
    pred_y = estimator.predict(validation_X)
    measure_acc(pred_y, validation_Y)

"""
replaces a column in the dataframe with 
a onehot encoding of it.
"""
def col_onehot_encoding(col):
    global data
    prefix = col.lower().replace(" ","_")
    data = pd.concat([data, pd.get_dummies(data[col], prefix=prefix)], axis=1)
    data.drop([col], axis=1, inplace=True)


"""
given predictions and real values this function prints the accuracy measures
saying how close the predictions are to the real values.
1. the percentage of predictions within the error range (percentage)
2. RMSE
3. MAE

@param y_pred - the vector containing the predictions
@param y - the vector containing the real values
"""
def measure_acc(y_pred, y, percentage=0.1):
    s = 0
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


"""
onehot encodes all columns who's name appear in the cols list.

@param cols - a list with column names to encode.
"""
def encode_onehot(cols):
    for col in cols:
        col_onehot_encoding(col)
    #print(data.dtypes)

"""
onehot encodes all columns who's name appear in the cols list.

@param cols - a list with column names to encode.
"""
def encode_category(cols):
    for col in cols:
        print(col)
        category_to_numbers(col)
    print(data.dtypes)


"""
rescales the numerical columns's values to fit better to the neural net
(in practice preprocessing had a negative result on the neural net's performance)
"""
def preprocess_data(train_data):
    global data
    normalized_cols = ["LAND SQUARE FEET", "GROSS SQUARE FEET", "TOTAL UNITS", "COMMERCIAL UNITS", "YEAR BUILT"\
                       , "SALE DATE", "lat", "lng", "homeless_center_dist", "museum_dist", "museum_cnt_3000",\
                       "theater_dist", "theater_in_brodway", "theaters_cnt_3000", "broadway_cnt_3000", "park_dist"\
                       ,"park_acres", "acres_cnt_3000"]
    norm_data = data[normalized_cols]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data[normalized_cols])
    rescaled_data = scaler.transform(norm_data)

    df = pd.DataFrame(rescaled_data)
    df.columns = normalized_cols

    usual_data = data.drop(normalized_cols, axis=1)
    data = pd.concat([df, usual_data], axis=1)




def main():
    global validation_X, validation_Y, test_X, test_Y, data
    cols = ["BOROUGH", "NEIGHBORHOOD", "BUILDING CLASS CATEGORY", "TAX CLASS AT PRESENT", "BUILDING CLASS AT PRESENT", \
            "BUILDING CLASS AT TIME OF SALE", "YEAR TYPE", "TAX CLASS AT TIME OF SALE"]

    encode_onehot(cols)

    print(data["SALE DATE"][0].split("/")[2][:4])
    data["SALE DATE"] = data["SALE DATE"].apply(lambda date: int(date.split("/")[2][:4])).astype(int)  # this is temporary

    #data["SALE DATE"] = data["SALE DATE"].apply(lambda date: int(date.split("-")[0])).astype(int)

    #preprocess_data(data)

    #print(data.iloc[0])

    #train, test and validation split)
    train_data, test_data = train_test_split(data, test_size=0.3)
    test_data, validation_data = train_test_split(test_data, test_size=0.5)
    print("data size:", data.shape)
    print("test size: ", test_data.shape)
    print("train size: ", train_data.shape)
    print("validation size: ", validation_data.shape)

    #remove unwanted fields from the train_cols (which will be used for train data later)
    train_cols = list(train_data.columns.values)
    train_cols.remove("SALE PRICE")
    #train_cols.remove("Unnamed: 0")
    train_cols.remove("ADDRESS")  # temporery
    train_cols.remove("APARTMENT NUMBER")  # temperary

    #train_data = train_data.head(500)

    X = train_data.loc[:, train_cols]
    Y = train_data.loc[:, ["SALE PRICE"]]


    validation_X = validation_data.loc[:, train_cols]
    validation_Y = validation_data.loc[:, ["SALE PRICE"]].astype(int)

    test_X = test_data.loc[:, train_cols]
    test_Y = test_data.loc[:, ["SALE PRICE"]].astype(int)

    #run the learning algorithms

    #run_xgboost(X,Y)
    #run_ridge_reg(X, Y)
    #run_svm(X, Y)/
    #run_knn(X, Y)
    run_nn(X, Y)
    #run_random_forest(X,Y)


main()
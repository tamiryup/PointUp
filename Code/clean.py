###### import ######
import pandas as pd
import numpy as np
import sklearn
#import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
import datetime

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
# import xgboost as xgb

from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

###### clean the data as explained in the documentation ######
def clean(data):
	data = data.drop_duplicates(data.columns, keep='last')
	data = data[data["SALE PRICE"]<4000000]
	data = data[data["GROSS SQUARE FEET"]<20000]
	data = data[data["LAND SQUARE FEET"]<25000]
	data = data[data["YEAR BUILT"]>1750]
	data = data[data["TAX CLASS AT PRESENT"]!="4"]
	data = data.sample(frac=1).reset_index(drop=True)
	data = data.drop(columns=["RESIDENTIAL UNITS"])
	data = data.drop(columns=["EASE-MENT"])
	data = data.drop(columns=["ZIP CODE"])
	return data

###### run and save ######
print("loading")
data = pd.read_csv("trial.csv")
print("cleaning")
data = clean(data)
data.to_csv("trial_clean.csv")

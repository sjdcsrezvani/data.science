# over fitting is common mistake : l1 l2 regularization is solution to it
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

import warnings
warnings.filterwarnings('ignore')  # suppress warnings for clean notebook
dataset = pd.read_csv('C:\\Users\\PaRsAfZaR\\Downloads\\Melbourne_housing_FULL.csv')
dataset.nunique()  # prints number of unique values

# by visual observation we can drop some columns that we don't use
cols_to_use = ['Suburb','Rooms','Type','Method','SellerG','Regionname','Propertycount','Distance',
               'CouncilArea','Bedroom2','Bathroom','Car','Landsize','BuildingArea','Price']  # columns we want to use
dataset = dataset[cols_to_use]  # now we have dataset with columns we want to use
dataset.isna().sum()  # it will print the total number of NaN values in each column
cols_to_fill_zero = ['Propertycount','Distance','Bedroom2','Bathroom','Car']  # fill NaN values in these columns with
# zero
dataset[cols_to_fill_zero] = dataset[cols_to_fill_zero].fillna(0)

dataset['Landsize'] = dataset['Landsize'].fillna(dataset.Landsize.mean())
dataset['BuildingArea'] = dataset['BuildingArea'].fillna(dataset['BuildingArea'].mean())  # fill NaN values in these
# columns with mean value of the columns

# when we have few NaN values in some columns we use dropna function which will drop those values
# now we have to convert any string value into categorized values
dataset = pd.get_dummies(dataset, drop_first=True)
# now we create x and y for our model
x = dataset.drop('Price', axis=1)
y = dataset['Price']
from sklearn.model_selection import train_test_split as tts
train_x , test_x , train_y , test_y = tts(x, y, test_size=0.3, random_state=2)
from sklearn.linear_model import LinearRegression as lr
reg = lr().fit(train_x, train_y)  # we can train model in one line
reg.score(test_x, test_y)  # our score will be very low 0.13
reg.score(train_x,train_y) # it will give score on training dataset 0.68 and when we have such a distance between
# train and test score it means it does over fitting
# there is lasso function for this matter
from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=50, max_iter=100, tol=0.1)  # read the parameters in lasso function
lasso_reg.fit(train_x,train_y)  # it will fit the model with regularization parameter on
lasso_reg.score(test_x,test_y)  # it will give closer score to train dataset score
lasso_reg.score(train_x, train_y) # its L1 regularization

# L2 regularization:
from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=50, max_iter=100, tol=0.1)
ridge_reg.fit(train_x,train_y)
ridge_reg.score(test_x,test_y)  # it does also give us the better score


# machine learning is set of techniques to extract knowledge from available data and use that knowledge to make
# decisions : 1) deep learning   2) mathematical models
# linear regression model : it will find a linear equation line in our chart that data points have minimum distance with
# linear equation :  lets say we have dataset of prices and areas : price=m* area + b
# m = slope(shib) , b = Y-axis intercept

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model  # importing the machinelearning lib

df = pd.read_csv('C:\\Users\\PaRsAfZaR\\Downloads\\homeprices.csv')
plt.scatter(df.area, df.price)  # it will plot the scatter chart (parakandegi)
reg = linear_model.LinearRegression()  # create an object for linear regression model
reg.fit(df[['area']], df.price)  # now our model will train based on data we are giving to it

# logistic regression: in linear regression we could predict dataframes like (home prices , weather , stock prices)
# and the predicted value was continuous

# in classification problems we will calculate dataframes like ( 1)email is spam  or not , 2) will customer buy life
# ensurance or not , 3) which party a person is going to vote for) , predicted value is categorical
# logistic regression is one of the techniques used for classification

# classification types: 1) the outcome is either yes or no , or True or false (binary classification)
# 2) the outcome is one of the categorical answers (multiclass classification)
# in logistic regression we won't draw linear line instead we draw logistic line (line will have curves)(logit function)
# logit function or sigmoid:  sigmoid(z) = 1/(1+e**(-z))   , e= euler's number
# sigmoid function converts inputs into range 0 to 1 ,  line we get looks like 'S' shape
# formula for linear regression is: y = m * x + b
# formula for logistic regression is: y = 1 / (1 + e ** -(m * x + b))

from sklearn.model_selection import train_test_split  # this function will split our data into train part and test part

# it will train the model based on train part, and then we can test the model based on test part
x_train, x_test, y_train, y_test = train_test_split(df[['area']], df.price, test_size=0.1)  # it will take 90% of
# the size for train part and 10% for test part , and it will give us 4 parameters
# it will choose our parts randomly not by order

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()  # creating our logistic regression model object
model.fit(x_test, y_train)  # training our model based on train part
# now with x_test part we will do the prediction and compare it with y_test
model.score(x_test, y_test)  # it will calculate the accuracy of the model by testing x_test and y_test

# multiclass classification:
# identify handwritten digits recognition:
import matplotlib as plt
#  %matplotlib inline
from sklearn.datasets import load_digits  # sklearn have alot of datasets which is used for training the model and

# we load digits dataset
digits = load_digits()
dir(digits)  # create our training set which is contain : 'data' , 'images' , 'target' , 'target_names'
# , 'feature_names', 'frame'
digits.data[0]  # prints the first data , data value
plt.gray()
plt.matshow(digits.images[0])  # it will plot the first data image
digits.target[0]  # it will show the target that corresponding the first image in our digits' dataset which is zero

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)  # we create test , train parts for
# our dataset with test size 20%
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs', max_iter=10000)  # create our logistic regression model object
# we have set iteration to the higher amounts for big datasets
model.score(x_test,y_test)  # calculate the accuracy of our model
plt.matshow(digits.images[67])  # plotting the random number image in our dataset
digits.target[67] # prints the target of our random number
model.predict([digits.data[67]])  # now will predict the target based on our model ,  we pass the data because our
# image is binary data and our data column is numeric data which is better for our model

# now we want to know when our model fails : using confusion matrix
y_predicted = model.predict(x_test)  # get all the predicted values for the x_test
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)  # we pass the true values and predicted values and get the 2*2 array
# now we want to visualize this array
import seaborn as sn  # importing seaborn for plotting the array

plt.figure(figsize = (10,7))  # creating figure or frame for (10,7) pixels size
sn.heatmap(cm ,annot=True)  # it will show model failure in heat map plot
plt.xlabel('predicted')
plt.ylabel('truth')
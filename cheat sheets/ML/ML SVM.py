# we use iris flower dataset like before and scatter plot the petal length and petal width to identify setosa ot
# vesicolor we can draw many lines to separate the datapoints but which is better we can measure the distance between
# nearby datapoints and the line which is called margin bigger the margin it's better for our model , support vector
# machine(SVM) SVM draws a hyper plane in n dimensional space such that it maximizes margin between classification
# groups when we draw curve line between datapoints with minimum margin in terms of nearby datapoints : high gamma
# when we draw straight line calculating all datapoints margin : low gamma : might get some inaccurate result when we
# try to use high gamma method and over fit the data : high regularization (C) when we try ti use low gamma method ,
# we might get some error : low regularization(C) when we have complex data, and we can't do any classification ,
# we add new dimension then we try to do classification with new dimension: this transformation called kernel
import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris() # loading iris dataset
df = pd.DataFrame(iris.data, columns=iris.feature_names)  # turning our data into dataframe
df['target'] = iris.target  # adding target data into our dataframe
df['folower_name'] = df.target.apply(lambda x: iris.target_names[x])  # adding target names to dataframe**

from matplotlib import pyplot as plt
df0 = df[df.target==0]
df1 = df[df.target==1]
df2 = df[df.target==2]  # separate dataframe based on target
plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color='g', marker='+')
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color='r', marker='+')
plt.scatter(df2['sepal length (cm)'],df2['sepal width (cm)'],color='b', marker='+')
# plot scatter for all dataframes for sepal length and width

plt.scatter(df0['petal length (cm)'],df0['petal width (cm)'],color='g', marker='+')
plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'],color='r', marker='+')
plt.scatter(df2['petal length (cm)'],df2['petal width (cm)'],color='b', marker='+')
# plot scatter for all dataframes for petal length and width


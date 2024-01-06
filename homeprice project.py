import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (20,10)

df1 = pd.read_csv('C:\\Users\\PaRsAfZaR\\Downloads\\bengaluru_house_prices.csv')
df1['price'] = df1['price'].apply(lambda x: int(x*1200))  # changing rupees into dollar
df1.groupby('area_type')['area_type'].agg('count')  # counting each element in area type column and show it
# now lets drop non-important columns
df2 = df1.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns')
# our important columns are in df2 now
df2.isnull().sum()  # it will show how many NaN values are in each column
# since our dataset is big and number of NaN values are small , we will drop them all
df3 = df2.dropna()
df3['size'].unique()  # it will give us every unique values in size column
df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))  # bedroom and bhk is the same, so we want only number part of the values , we split string into 2 and first part is numeric then we convert it into integer
# bhk (bedroom,hall,kitchen)
# now we check anything on bhk column which is not logical , here we have 43 bedrooms in 2400 square feet which is not logical

# in our square feet column we have values which is not integer, and we want to convert them
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True  # this will return value if its float and if it is not returns false
df3[~df3['total_sqft'].apply(is_float)]  # it will apply is_float function on total_sqft column, '~' it will show the false values instead of true values
# now we see that dataset total_sqft column has alot of errors
# anytime we see range we take average and ignore other error values
def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None

df4 = df3.copy()  # it creates a deep copy of dataframe
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)
df4['total_area'] = df4['total_sqft'].apply(lambda x: float(x*0.093))  # convert square feet into square meter
df4 = df4.drop('total_sqft', axis='columns')

df4['price_per_sqm'] = df4['price']/df4['total_area']  # adding column : price per square meter
len(df4.location.unique())  # number of unique values in location column
# we can't create dummies for each unique location because we are going to have too many features (columns) in dataframe
df4.location = df4.location.apply(lambda x: x.strip())  # it will clean the string values if there is any additional space
location_stats = df4.groupby('location')['location'].agg('count').sort_values(ascending=False)  # it will show the count of each location in order (higher to lower)
# now we can remove locations that are repeating few times
location_stats_less_than_10 = location_stats[location_stats<=10]  # saving locations that have less than 10 counts
# we can put all these locations into a general category called other
df4.location = df4.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)  # change these locations to other

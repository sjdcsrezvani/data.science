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

# now we do outlier removal:
# we don't have any house where area per bedroom is less than x amount, and we want to remove any outlier
# first reverse square meter to square foot
df4[df4.total_sqft/df4.bhk<300].head()  # there is unusual values and have to remove that
df5 = df4[~(df4.total_sqft/df4.bhk<300)] # now normal data is in df5

# now we want to see how many outliers are in price_per_sqft column, and we want to do it based on each location and categorize it
# we write a function that go through each location category and calculate std and mean then remove the outlier(we use 1 std for outlier detection)
def remove_pps_outlier(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df], ignore_index=True)
    return df_out

df6 = remove_pps_outlier(df5)  # now we removed outlier based on each location

# now we want to know how many houses with same amount of bhk and area have different prices
def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcparams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft, bhk2.price_per_sqft, color='b', label='2 bhk', s=50)
    plt.scatter(bhk3.total_sqft, bhk3.price_per_sqft, marker='+', color='g', label='3 bhk', s=50)
    plt.xlabel('total square feet area')
    plt.ylabel('price per square feet')
    plt.title(location)
    plt.legend()  # this will plot those houses based on location
plot_scatter_chart(df6,'Rajaji Nagar')
# as we can see there is some datapoints where in the same location with same area , 2 bhk is pricier than 3 bhk
# these are outliers, and we want to remove them
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats= {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqm),
                'std': np.std(bhk_df.price_per_sqm),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqm<(stats['mean'])].index.values)
    return df.drop(exclude_indices, axis='index')
df7 = remove_bhk_outliers(df6)
mpl.rcParams['figure.figsize'] = (20,10)
plt.hist(df7.price_per_sqm, rwidth=0.8)
plt.xlabel('price per square meter')
plt.ylabel('count')  # by plotting histogram we can see our data is normal distributed
# from 0 to 1500 price per square meter , there is the majority of our datapoints
plt.hist(df7.bath, rwidth=0.8)  # we can see most of the datapoints are in 0 to 6 bath
df8 = df7[df7.bath<df7.bhk+2]  # clear the datapoints where number of bathrooms is greater than number of bedrooms +2
df9 = df8.drop(['size','price_per_sqm'], axis='columns')  # drop non-important columns
# now our dataframe is pretty much clean and let's prepare it for our model

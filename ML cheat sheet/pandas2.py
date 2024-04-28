import pandas as pd
df = pd.DataFrame({
    'city':['new york','chicago','orlando'],
    'tempt':[21,14,35]
})
df1 = pd.DataFrame({
    'city': ['chicago','new york','orlando'],
    'humidity': [65,68,75]
})  # create data frame contains tempt and humidity

df3 = pd.merge(df,df1,on='city')  # it will merge two dataframe into one data frame based on column 'city'
# if we have missing data on some cities it wouldn't merge them, only cities that have all data
# it will be inner join them and (df^df1) means it will join columns that are shared in dataframes

df3 = pd.merge(df,df1,on='city', how='outer')  # now it will merge all elements
df3 = pd.merge(df,df1,on='city', how='left')  # it will join common elements and elements in first dataframe
df3 = pd.merge(df,df1,on='city', how='right')  # it will join common elements and elements in second dataframe
df3 = pd.merge(df,df1,on='city', how='outer')  # it will indicate elements come from which dataframe



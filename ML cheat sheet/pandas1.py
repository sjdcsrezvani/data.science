# these codes are for jupyter notebook
# lesson one
import pandas as pd
df = pd.read_csv('C:\\Users\\PaRsAfZaR\\Downloads\\weather.csv') # reads a csv file as data frame in df
df['max tempt'].max()  # prints a maximum number in max tempt column
df['date'][df['max tempt'] == 51]  # Print dates that the max tempt is 51

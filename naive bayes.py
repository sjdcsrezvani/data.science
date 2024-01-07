import pandas as pd
df = pd.read_csv('C:\\Users\\PaRsAfZaR\\Downloads\\titanic.csv')

df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'], axis='columns', inplace=True)
# drop those columns that doesn't matter to our model
target = df.Survived
inputs = df.drop(['Survived'], axis='columns')

# sex column is string valued, so we convert it to numeric:
dummies = pd.get_dummies(inputs.Sex)  # get dummy columns for our string values
inputs = pd.concat([inputs,dummies], axis='columns')  # merge it to inputs
inputs.drop('Sex', axis='columns', inplace=True)  # drop the sex column
inputs.columns[inputs.isna().any()]  # this will show us which column has NaN value
inputs.Age = inputs.Age.fillna(round(inputs.Age.mean()))  # fill the NaN values

from sklearn.model_selection import train_test_split as tts
x_train , x_test , y_train , y_test = tts(inputs,target , test_size= 0.3)  # create our train test parts

from sklearn.naive_bayes import GaussianNB as gs
model = gs()  # create gaussian naive bayes model : we use this when our data distribution is normal or bell curve

model.fit(x_train,y_train)
model.score(x_test,y_test)
model.predict_proba(x_test)  # it will give us the probability of not surviving or surviving for each value in x_test

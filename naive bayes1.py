# spam detection in emails:

import pandas as pd
df= pd.read_csv('C:\\Users\\PaRsAfZaR\\Downloads\\spam.csv')  # load the csv contains the emails which categorized in
# ham and spam

df.groupby('Category').describe()  # it will group the data by category and describe it

# we have to convert all string values to number somehow because we know models won't work with strings
df['spam'] = df['Category'].apply(lambda x: 1 if x=='spam' else 0)  # add new column spam = 1 and ham = 0

from sklearn.model_selection import train_test_split as tts
x_train , x_test , y_train , y_test = tts(df.Message,df.spam, test_size=0.25)

# we use count vectorised technique : will find out unique word in each message then count each one of them in messages
from sklearn.feature_extraction.text import CountVectorizer as cv
v = cv()
x_train_count = v.fit_transform(x_train.values)  # it will save the frequency of each unique word in our message

from sklearn.naive_bayes import MultinomialNB as mn  # we use this naive model when our features are binary (0,1)
model = mn()
model.fit(x_train_count,y_train)

emails = [
    'Hey mohan, can we get together to watch footbal game tomorrow?',
    'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!'
]
emails_count = v.transform(emails)
model.predict(emails_count)  # it will predict first one is ham and second one is spam

x_test_count = v.transform(x_test)
model.score(x_test_count,y_test)

from sklearn.pipeline import Pipeline as pl
clf = pl([
    ('vectorizer', cv()),
    ('nb', mn())
])  # create our pipeline to convert any string to vector number : this code is equal to all code above

clf.fit(x_train,y_train)  # now we can train model directly on x_train which is string valued

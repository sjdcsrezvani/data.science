# decision tree:
# when our data is too scattered decision tree algorithm will divide dataset and decide matters based on that
# it will select each attribute and go on
# select ordering of features:
# select based on which way , at least one of the samples has minimum entropy
import pandas as pd
df = pd.read_csv('C:\\Users\\PaRsAfZaR\\Downloads\\salaries.csv')

inputs = df.drop('salary_more_then_100k', axis='columns')
target = df['salary_more_then_100k']  # divide our dataset to independent(input) variables and target variables

from sklearn.preprocessing import LabelEncoder  # we label our inputs dataset because it has string data
le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()  # create label object for each column
inputs['company_n'] = le_company.fit_transform(inputs['company'])
inputs['job_n'] = le_job.fit_transform(inputs['job'])
inputs['degree_n'] = le_degree.fit_transform(inputs['degree'])  # it will train the objects and scale them based on
# column and create new one
inputs_n = inputs.drop(['company','job','degree'],axis ='columns')  # now we drop those label columns and now we have
# all numeric dataset

from sklearn import tree
model = tree.DecisionTreeClassifier()  # create our tree classifier object model
model.fit(inputs_n,target)  # train our model based on input_n and target
model.score(inputs_n,target)  # score is going to be one because we are using th same dataset
model.predict([[2,2,1]])

# evaluating model performance: we have several options to do it: 1) use all available data for training and test on
# same dataset when we want to predict a data similar to our dataset (not a good way)
# 2) split available dataset into training and test sets ( good way to validate our model)
# 3) k fold cross validation : divide our dataset samples into folds : pick the first fold for testing and
# the rest for training then calculate the score , second time we pick second fold for test and take the score
# we do the same for other stages : pick fold(n) for test and take the score
# then we take average of all the scores we got (perfect technique for validating our model)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.datasets import load_digits

digits = load_digits()  # loading the digits dataset
x_train , x_test , y_train , y_test = tts(digits.data,digits.target,test_size=0.3)

lr = LogisticRegression(solver='sag',max_iter=900)
lr.fit(x_train,y_train)
lr.score(x_test,y_test)  # we create logistic regression classifier and take the score

svm = SVC()
svm.fit(x_train,y_train)
svm.score(x_test,y_test)  # create model based on svm and take score

rf = RandomForestClassifier(n_estimators=40)
rf.fit(x_train,y_train)
rf.score(x_test,y_test)  # create model based on random forest and take score
# this is a quick way to compare the score between algorithms

# now lets try k fold:
from sklearn.model_selection import KFold
kf = KFold(n_splits=3)  # creating k fold model object with 3 folds
def get_score(model,x_train , x_test , y_train , y_test):
    model.fit(x_train,y_train)
    return model.score(x_test,y_test)  # this is very important, we can pass any model, and it will return the score

from sklearn.model_selection import StratifiedKFold as sfk  # when we have dataset with too many classification
# we use this model
folds = sfk(n_splits=3)

scores_l = []
scores_svm = []
scores_rf = []

for train_index, test_index in kf.split(digits.data):
    x_train, x_test, y_train, y_test = digits.data[train_index], digits.data[test_index], digits.target[train_index], \
                                       digits.target[test_index]
    print(get_score(LogisticRegression(),x_train , x_test , y_train , y_test))
    print(get_score(SVC(),x_train , x_test , y_train , y_test))
    print(get_score(RandomForestClassifier(),x_train , x_test , y_train , y_test))
# it will print every fold score on each model
# we can do this code for the codes above:
from sklearn.model_selection import cross_val_score as cvs
cvs(LogisticRegression(), digits.data, digits.target)  # we pass the model and our x and y


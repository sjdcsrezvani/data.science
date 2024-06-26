import numpy as np
import pywt
import cv2


# we do wavelet transformation

def w2d(img, mode='haar', level=1):  # this function take an image and return a new image that is wavelet transformed
    imarray = img  # datatype conversions
    imarray = cv2.cvtColor(imarray, cv2.COLOR_RGB2GRAY)  # convert to gray
    imarray = np.float32(imarray)  # convert to float
    imarray /= 255;
    coefs = pywt.wavedec2(imarray, mode, level=level)  # compute coefficients

    coefs_h = list(coefs)
    coefs_h[0] *= 0;  # process coefficients

    # reconstruction
    imarray_h = pywt.waverec2(coefs_h, mode);
    imarray_h *= 255;
    imarray_h = np.uint8(imarray_h)

    return imarray_h  # it will turn our image into more readable image for computer and give it more details


im_har = w2d(cropped_image, 'db1', 5)
plt.imshow(im_har, cmap='gray')  # this will show image that wavelet transformed

celebrity_file_names_dict = {}  # since we deleted some useless images manually , remap the dictionary with new dataset
for img_dir in cropped_image_dirs:
    celebrity_name = img_dir.split('\\')[-1]
    file_list = []
    for entry in os.scandir(img_dir):
        file_list.append(entry.path)
    celebrity_file_names_dict[celebrity_name] = file_list

class_dict = {}
count = 0
for celebrity_name in celebrity_file_names_dict.keys():
    class_dict[celebrity_name] = count
    count = count + 1  # creating dictionary and assign a number to each celebrity name

# raw image and wavelet transformed image are inputs of our classifier, and we have to stack them on each other

x, y = [], []  # creating x and y for our model training

for celebrity_name, training_files in celebrity_file_names_dict.items():
    for training_image in training_files:
        img = cv2.imread(training_image)  # reading every image in our dictionary
        # if img is None: # we deleted some images , so some directories return none, this will be error, so we have to
        # implement the condition if there is no image then continue the program, after remapping the dictionary
        # we don't have use this
        # continue
        scalled_raw_img = cv2.resize(img, (32, 32))  # resize them to make sure everything are same size
        img_har = w2d(img, 'db1', 5)  # we do wavelet transformation
        scalled_img_har = cv2.resize(img_har, (32, 32))  # then scale it too
        combined_image = np.vstack(
            (scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))  # combine 2
        # images together vertically
        x.append(combined_image)  # combined images are x, input of the model
        y.append(class_dict[celebrity_name])  # celebrity names are y, output of the model

x = np.array(x).reshape(len(x), 4096).astype(float)  # convert x to an array and turn data into float

# now our data is ready for model training

# import sklearn moduls that we are using
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)  # split our data into train and test

pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel='rbf', C=10))])  # we want to scale our dataset,
# it will first scale it base on standardscaler then create svm model with certain parameters

pipe.fit(x_train, y_train)  # train our scaled model
pipe.score(x_test, y_test)

print(classification_report(y_test, pipe.predict(x_test)))  # it will compare our model result with valid data
# and give us the reports  # read about f1 score

# now we try grid search cv to find the best model and parameters

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV


model_params = {  # save our parameters we want to tune into dictionary
    'svm': {
        'model': svm.SVC(gamma='auto', probability=True),
        'params': {
            'svc__C': [1,10,100,1000],
            'svc__kernel': ['rbf','linear']
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'randomforestclassifier__n_estimator': [1,5,10]
        }
    },
    'logistic_regression': {
        'model': LogisticRegression(solver='liblinear', multi_class='auto'),
        'params': {
            'logisticregression__C': [1,5,10]
        }
    }
}

scores = []
best_estimators = {}

import pandas as pd

for algo, mp in model_params.items():
    pipe = make_pipeline(StandardScaler(), mp['model'])  # create pipeline because we have to scale the data
    clf = GridSearchCV(pipe, mp['params'], cv=5, return_train_score=False)  # using grid search cv to create our model
    clf.fit(x_train, y_train)
    scores.append({  # appending best score and params into scores list
        'model': algo,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    best_estimators[algo] = clf.best_estimator_

df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])  # saving the result into dataframe to show
# it
# here logistic regression has best score with params: 'C=1'

best_clf = best_estimators['logistic_regression']  # save our best params and model into best_clf

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, best_clf.predict(x_test))  # using confusion matrix will tell us errors in our model
# we can use seaborn to visualize that
import seaborn as sn
plt.figure(figsize= (10,7))
sn.heatmap(cm, annot=True)

# It's time to save our model
import joblib
joblib.dump(best_clf, 'saved_model.pkl')  # save it as pickle file

import json
with open('class_dictionary.json','w') as f:
    f.write(json.dumps(class_dict))  # saving our class dictionary with json


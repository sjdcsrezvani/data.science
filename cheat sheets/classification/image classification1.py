import numpy as np
import cv2  # we can read images
import matplotlib
from matplotlib import pyplot as plt

# %matplotlib inline

img = cv2.imread('G:\\celebrity image classification\\test_images\\sharapova1.jpg')  # read the image with cv2
img.shape  # image has 3 dimension shape , third one is rgb indicator

plt.imshow(img)  # showing image with matplotlib
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # we can use this function to gray our image , now our image only has 2
# dimension
gray.shape
plt.imshow(gray, cmap='gray')  # showing gray image

# we are using haarcascades classification to detect eyes and faces
face_cascade = cv2.CascadeClassifier('G:\\celebrity image '
                                     'classification\\haarcascades\\haarcascade_frontalface_default.xml')  # loading
# front face classifier
eye_cascade = cv2.CascadeClassifier('G:\\celebrity image classification\\haarcascades\\haarcascade_eye.xml')  # loading
# eye classifier

faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # it will detect faces in our gray image
(x, y, w, h) = faces[0]  # x and y is a point where our face is starting and h is height and w is width

face_img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
plt.imshow(face_img)  # now that we detect faces we can pass the points to matplotlib and show it

cv2.destroyAllWindows()  # it will destroy everything that we have done
for (x, y, w, h) in faces:  # this code detect each faces and for each face it will detect eyes then draw rectangle
    face_img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = face_img[y:y + h, x:x + w]  # roi is region of interest , cropped face
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

plt.figure()
plt.imshow(face_img, cmap='gray')
plt.show()  # now we can show image with face and eyes detected


# now we can write a function to take the image and returns a cropped face

def get_cropped_image_if_2_eyes(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:  # if our face has 2 eyes in our image then return it
            return roi_color


orginal_image = cv2.imread('G:\\celebrity image classification\\test_images\\sharapova1.jpg')
cropped_image = get_cropped_image_if_2_eyes('G:\\celebrity image classification\\test_images\\sharapova1.jpg')  # it
# will give us face image using a function
# if 2 eyes are not visible then it will return nothing

path_to_data = 'G:\\celebrity image classification\\dataset'  # save our directory in variables
path_to_cr_data = 'G:\\celebrity image classification\\dataset\\cropped'

import os

img_dirs = []  # it will save all subdirectory into this
for entry in os.scandir(path_to_data):
    if entry.is_dir():
        img_dirs.append(entry.path)

import shutil
if os.path.exists(path_to_cr_data):  # if cropped folder doesn't exist we are going to create it
    shutil.rmtree(path_to_cr_data)
os.mkdir(path_to_cr_data)




cropped_image_dirs = []
celebrity_file_names_dict = {}

for img_dir in img_dirs:
    count = 1
    celebrity_name = img_dir.split("\\")[-1]  # it will split the path of each directory to get celebrity names
    celebrity_file_names_dict[celebrity_name] = []
    for entry in os.scandir(img_dir):
        roi_color = get_cropped_image_if_2_eyes(entry.path)  # it will pass each image to the cascade function we wrote
        if roi_color is not None:
            cropped_folder = path_to_cr_data + celebrity_name
            if not os.path.exists(cropped_folder):
                os.makedirs(cropped_folder)
                cropped_image_dirs.append(cropped_folder)

            cropped_file_name = celebrity_name + str(count) + '.png'
            cropped_file_path = cropped_folder + '\\' + cropped_file_name

            cv2.imwrite(cropped_file_path, roi_color)  # write cropped images into each celebrity folders
            celebrity_file_names_dict[celebrity_name].append(cropped_file_path)  # save all cropped images paths into
            # dictionary
            count += 1

# there is cropped images of other people in our dataset, we have to clean them manually

#!/usr/bin/env python
# coding: utf-8

# In[2]:



import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder 
import keras 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from PIL import Image

data4 = [] #for storing image data in numpy array form
paths = [] #for storing paths of all images
result5 = [] #for storing one hot encoded form of target class whether normal or tumor

for r, d, f in os.walk(r'../input/brain-mri-images-for-brain-tumor-detection/yes'):
    for file in f:
        if '.jpg' in file:
            paths.append(os.path.join(r, file)) #getting input data from directory
            
print("The number of MRI Images labelled 'yes':",len(os.listdir(r'../input/brain-mri-images-for-brain-tumor-detection/yes'))) #getting to know the number of mri images labelled yes

encode = OneHotEncoder()
encode.fit([[0], [1]])  # for converting the target class into 1s and 0s  0 - Tumor,1 - Normal

for path in paths:
    images = Image.open(path)
    images = images.resize((128,128))
    images = np.array(images) #preprocessing
    if(images.shape == (128,128,3)):
        data4.append(np.array(images))
        result5.append(encoder.transform([[0]]).toarray())
# This cell updates result list for images without tumor

paths = []
for r, d, f in os.walk(r"../input/brain-mri-images-for-brain-tumor-detection/no"):
    for file in f:
        if '.jpg' in file:
            paths.append(os.path.join(r, file))
            
print("The number of MRI Images labelled 'no':",len(os.listdir(r'../input/brain-mri-images-for-brain-tumor-detection/no')))#getting to know the number of mri images labelled No

for path in paths:
    images = Image.open(path)
    images = images.resize((128,128)) #preprocessing
    images = np.array(images)
    if(images.shape == (128,128,3)):
        data4.append(np.array(images))
        result5.append(encoder.transform([[1]]).toarray()) #lists updated with images of tumor


result5 = np.array(result5) #array of the target variables encoded
result5 = result5.reshape(139,2)

data4 = np.array(data4)
data4.shape #input of data(images) in the form of numpy array


x_train,x_test,y_train,y_test = train_test_split(data4,result5, test_size=0.2, shuffle=True, random_state=0) #splitting the data into train/test split for modelling


model = Sequential()

model.add(Conv2D(32, kernel_size=(2, 2), input_shape=(128, 128, 3), padding = 'Same'))
model.add(Conv2D(32, kernel_size=(2, 2),  activation ='relu', padding = 'Same'))


model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size = (2,2), activation ='relu', padding = 'Same'))
model.add(Conv2D(64, kernel_size = (2,2), activation ='relu', padding = 'Same'))

model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss = "categorical_crossentropy", optimizer='Adamax')

print(model.summary()) #training the model and building CNN

y_train.shape

history = model.fit(x_train, y_train, epochs = 30, batch_size = 40, verbose = 1,validation_data = (x_test, y_test)) #fitting the trained model

y_pred = model.predict_classes(x_test)
#real values for test images
y_test_=np.argmax(y_test, axis=1)

from sklearn.metrics import classification_report

print("Classification report : \n",classification_report(y_test_, y_pred)) #gives you the complete report of of all the evaluation metrics

def prediction(result):
    if result==0:
        return 'Its a Tumor'
    else:
        return 'No, Its not a tumor'
    
from matplotlib.pyplot import imshow
img = Image.open(r"../input/brain-mri-images-for-brain-tumor-detection/no/N17.jpg")
x = np.array(img.resize((128,128)))
x = x.reshape(1,128,128,3)
res = model.predict_on_batch(x)
classification = np.where(res == np.amax(res))[1][0]
imshow(img)
print(str(res[0][classification]*100) + '% Accurate This Is ' + prediction(classification)) #prediction of a single image for no

from matplotlib.pyplot import imshow
img = Image.open(r"../input/brain-mri-images-for-brain-tumor-detection/yes/Y5.jpg") 
x = np.array(img.resize((128,128)))
x = x.reshape(1,128,128,3)
res = model.predict_on_batch(x)
classification = np.where(res == np.amax(res))[1][0]
imshow(img)
print(str(res[0][classification]*100) + '% accurate This Is A ' + prediction(classification)) #prediction of a single image for yes



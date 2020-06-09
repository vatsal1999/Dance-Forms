# -*- coding: utf-8 -*-
"""
Created on Tue May 26 09:22:20 2020

@author: admin
"""

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tqdm import tqdm

train = pd.read_csv('./train.csv')


train_image = []
for i in tqdm(range(train.shape[0])):
    img = image.load_img('Train/'+train['Image'][i], target_size=(128,128,3), grayscale=False)
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
X = np.array(train_image)



y=train['target']

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)

Y=y
Y = to_categorical(Y)

model = Sequential()
model.add(Conv2D(128, kernel_size=(3, 3),activation='relu',input_shape=(128,128,3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=0.2)

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

test_file = pd.read_csv('test.csv')

test_image = []
for i in tqdm(range(test_file.shape[0])):
    img = image.load_img('Test/'+test_file['Image'][i], target_size=(64,64,3), grayscale=False)
    img = image.img_to_array(img)
    img = img/255
    test_image.append(img)
test = np.array(test_image)

prediction = model.predict_classes(test)

prediction=le.inverse_transform(prediction)

sample = pd.read_csv('test.csv')
sample['Image'] = test_file['Image']
sample['target'] = prediction
sample.to_csv('test.csv', header=True, index=False)


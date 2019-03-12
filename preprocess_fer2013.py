# create data and label for FER2013
# labels: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
import csv
import os
import numpy as np
import h5py
import cv2
from ImageEnhance import *

file = 'data/fer2013.csv'

# Creat the list to store the data and label information
Training_x = []
Training_y = []
PublicTest_x = []
PublicTest_y = []
PrivateTest_x = []
PrivateTest_y = []

datapath = os.path.join('data', 'data.h5')
if not os.path.exists(os.path.dirname(datapath)):
    os.makedirs(os.path.dirname(datapath))

with open(file,'r') as csvin:
    data=csv.reader(csvin)
    for row in data:
        if row[-1] == 'Training':
            temp_list = []
            #reshape((48, 48))

            for pixel in row[1].split( ):
                temp_list.append(int(pixel))
            I = np.asarray(temp_list)
            test_image = I.reshape((48, 48))
            test_image = test_image.astype(np.uint8)
            test_image = test_image[:, :, np.newaxis]
            test_image = np.concatenate((test_image, test_image, test_image), axis=2)
            img_brightness = LuEnhance(test_image)

            img_gray = cvtColor(img_brightness, CV_BGR2GRAY)
            I1 = img_gray.reshape((1, 48*48))
            I2 = I1.reshape((-1))

            Training_y.append(int(row[0]))
            Training_x.append(I.tolist())

        if row[-1] == "PublicTest" :
            temp_list = []
            for pixel in row[1].split( ):
                temp_list.append(int(pixel))
            I = np.asarray(temp_list)
            PublicTest_y.append(int(row[0]))
            PublicTest_x.append(I.tolist())

        if row[-1] == 'PrivateTest':
            temp_list = []
            for pixel in row[1].split( ):
                temp_list.append(int(pixel))
            I = np.asarray(temp_list)

            PrivateTest_y.append(int(row[0]))
            PrivateTest_x.append(I.tolist())

print(np.shape(Training_x))
print(np.shape(PublicTest_x))
print(np.shape(PrivateTest_x))

datafile = h5py.File(datapath, 'w')
datafile.create_dataset("Training_pixel", dtype = 'uint8', data=Training_x)
datafile.create_dataset("Training_label", dtype = 'int64', data=Training_y)
datafile.create_dataset("PublicTest_pixel", dtype = 'uint8', data=PublicTest_x)
datafile.create_dataset("PublicTest_label", dtype = 'int64', data=PublicTest_y)
datafile.create_dataset("PrivateTest_pixel", dtype = 'uint8', data=PrivateTest_x)
datafile.create_dataset("PrivateTest_label", dtype = 'int64', data=PrivateTest_y)
datafile.close()

print("Save data finish!!!")

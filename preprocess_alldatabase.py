# create data and label for FER2013
# labels: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
import csv
import os
import numpy as np
import h5py
import cv2
from ImageEnhance import *

dst_resizelength = 100
datapath = os.path.join('data', 'data_wild.h5')

expression_dict = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Sad': 4, 'Surprise': 5, 'Neutral': 6}

expression_ind_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

expression_list = {'Angry': [], 'Disgust': [], 'Fear': [], 'Happy': [], 'Sad': [], 'Surprise': [], 'Neutral': []}

expression_count = {'Angry': 0, 'Disgust': 0, 'Fear': 0, 'Happy': 0, 'Sad': 0, 'Surprise': 0, 'Neutral': 0}

subpath = '0'
animate_label_list = []


# Creat the list to store the data and label information
Training_x = []
Training_y = []
PublicTest_x = []
PublicTest_y = []
PrivateTest_x = []
PrivateTest_y = []

def xrange(x):
    return iter(range(x))


def processFER2013ToImage(base_path, flag):
    file = 'data/fer2013.csv'
    with open(file, 'r') as csvin:
        data = csv.reader(csvin)
        for row in data:
            if row[-1] == flag:
                temp_list = []
                # reshape((48, 48))
                expression_str = expression_ind_dict[int(row[0])]
                expression_count[expression_str] += 1
                file_str = os.path.join(base_path+expression_str, str(expression_count[expression_str])+'.jpg')
                file_str1 = os.path.join(base_path + expression_str, str(expression_count[expression_str]) + '_origin.jpg')

                for pixel in row[1].split():
                    temp_list.append(int(pixel))
                I = np.asarray(temp_list)
                test_image = I.reshape((48, 48))
                test_image = test_image.astype(np.uint8)
                test_image = test_image[:, :, np.newaxis]
                test_image = np.concatenate((test_image, test_image, test_image), axis=2)
                test_image1 = test_image
                cv2.imwrite(file_str1, test_image1)
                img_brightness = LuEnhance(test_image)
                test_image1 = test_image1.astype(np.uint8)

                img_brightness = img_brightness.astype(np.uint8)

                cv2.imwrite(file_str, img_brightness)


def processFER2013():
    file = 'data/fer2013.csv'
    # if not os.path.exists(os.path.dirname(datapath)):
    #     os.makedirs(os.path.dirname(datapath))
    with open(file, 'r') as csvin:
        data = csv.reader(csvin)
        for row in data:
            if row[-1] == 'Training':
                temp_list = []
                # reshape((48, 48))

                for pixel in row[1].split():
                    temp_list.append(int(pixel))
                I = np.asarray(temp_list)
                test_image = I.reshape((48, 48))
                test_image1 = test_image
                test_image = test_image.astype(np.uint8)
                test_image = test_image[:, :, np.newaxis]
                test_image = np.concatenate((test_image, test_image, test_image), axis=2)
                img_brightness = LuEnhance(test_image)
                test_image1 = test_image1.astype(np.uint8)

                img_brightness = img_brightness.astype(np.uint8)

                img_brightnessBig = cv2.resize(img_brightness, (dst_resizelength, dst_resizelength))
                # cv2.imshow('test_image', test_image1)
                # cv2.imshow('img_brightness', img_brightness)
                # cv2.imshow('img_brightnessBig', img_brightnessBig)
                # cv2.waitKey(0)
                img_gray = cv2.cvtColor(img_brightnessBig, cv2.COLOR_BGR2GRAY)
                I1 = img_gray.reshape((1, dst_resizelength * dst_resizelength))
                I = I1.reshape((-1))

                Training_y.append(int(row[0]))
                Training_x.append(I.tolist())

            if row[-1] == "PublicTest":
                temp_list = []
                for pixel in row[1].split():
                    temp_list.append(int(pixel))
                I = np.asarray(temp_list)

                test_image = I.reshape((48, 48))
                test_image = test_image.astype(np.uint8)
                test_image = test_image[:, :, np.newaxis]
                test_image = np.concatenate((test_image, test_image, test_image), axis=2)
                img_brightness = LuEnhance(test_image)

                img_brightness = img_brightness.astype(np.uint8)

                img_brightnessBig = cv2.resize(img_brightness, (dst_resizelength, dst_resizelength))

                # cv2.imshow('img_brightnessBig', img_brightnessBig)
                # cv2.waitKey(0)
                img_gray = cv2.cvtColor(img_brightnessBig, cv2.COLOR_BGR2GRAY)
                I1 = img_gray.reshape((1, dst_resizelength * dst_resizelength))
                I = I1.reshape((-1))

                PublicTest_y.append(int(row[0]))
                PublicTest_x.append(I.tolist())

            if row[-1] == 'PrivateTest':
                temp_list = []
                for pixel in row[1].split():
                    temp_list.append(int(pixel))
                I = np.asarray(temp_list)

                test_image = I.reshape((48, 48))
                test_image = test_image.astype(np.uint8)
                test_image = test_image[:, :, np.newaxis]
                test_image = np.concatenate((test_image, test_image, test_image), axis=2)
                img_brightness = LuEnhance(test_image)
                # test_image = test_image.astype(np.uint8)
                img_brightness = img_brightness.astype(np.uint8)
                img_brightnessBig = cv2.resize(img_brightness, (dst_resizelength, dst_resizelength))
                # cv2.imshow('test_image', test_image)
                # cv2.imshow('img_brightness', img_brightness)
                # cv2.waitKey(0)
                img_gray = cv2.cvtColor(img_brightnessBig, cv2.COLOR_BGR2GRAY)
                I1 = img_gray.reshape((1, dst_resizelength * dst_resizelength))
                I = I1.reshape((-1))

                PrivateTest_y.append(int(row[0]))
                PrivateTest_x.append(I.tolist())

def processAnimate(base_path):
    global subpath
    global expression_list
    if not os.path.isdir(base_path) and not os.path.isfile(base_path):
        return False
    if os.path.isfile(base_path):
        file_path = os.path.split(base_path)
        lists = file_path[1].split('.')
        file_ext = lists[-1]
        img_ext = ['bmp', 'jpeg', 'gif', 'psd', 'png', 'jpg', 'tiff']
        if file_ext in img_ext:
            subpath_aplit = subpath.split('_')
            for key in expression_dict.keys():
                if subpath_aplit[-1] == key:
                    animate_label_list.append(expression_dict[subpath_aplit[-1]])
                    temp = cv2.imread(base_path)
                    img_size = temp.shape
                    if(img_size[0] == 0) or (img_size[1] == 0) or (img_size[2] == 0):
                        print('temp.cols == 0')
                    img_resize = cv2.resize(temp, (dst_resizelength, dst_resizelength))
                    img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)

                    I_gray = img_gray.reshape((1, dst_resizelength * dst_resizelength))
                    I_color = img_resize.reshape((1, 3 * dst_resizelength * dst_resizelength))
                    I = I_gray.reshape((-1))
                    expression_list[subpath_aplit[-1]].append(I.tolist())
                    break

    elif os.path.isdir(base_path):
        for x in os.listdir(base_path):
            cur_string = os.path.join(base_path, x)
            if os.path.isdir(cur_string):
                subpath = x
            processAnimate(cur_string)

def ReRangeTrainData(expression_key):
    img_list_len = len(expression_list[expression_key])

    train_label = [expression_dict[expression_key] for p in range(img_list_len)]

    Training_y.extend(train_label)

    Training_x.extend(expression_list[expression_key])

def ReRangeValidateData(expression_key):
    img_list_len = len(expression_list[expression_key])

    validate_label = [expression_dict[expression_key] for p in range(img_list_len)]

    PublicTest_y.extend(validate_label)

    PublicTest_x.extend(expression_list[expression_key])

def ReRangeTestData(expression_key):
    img_list_len = len(expression_list[expression_key])

    test_label = [expression_dict[expression_key] for p in range(img_list_len)]

    PrivateTest_y.extend(test_label)

    PrivateTest_x.extend(expression_list[expression_key])

def ReRangeData(expression_key):
    img_list_len = len(expression_list[expression_key])
    training_index = int(img_list_len * 0.8)
    validate_index = int((img_list_len-int(img_list_len * 0.8))/2+training_index)
    test_length = img_list_len-validate_index
    validate_length = img_list_len - training_index - test_length
    train_list = expression_list[expression_key][:training_index]
    validate_list = expression_list[expression_key][training_index:validate_index]
    test_list = expression_list[expression_key][validate_index:]

    train_label = [expression_dict[expression_key] for p in range(training_index)]
    validate_label = [expression_dict[expression_key] for p in range(validate_length)]
    test_label = [expression_dict[expression_key] for p in range(test_length)]

    Training_y.extend(train_label)
    PublicTest_y.extend(validate_label)
    PrivateTest_y.extend(test_label)

    Training_x.extend(train_list)
    PublicTest_x.extend(validate_list)
    PrivateTest_x.extend(test_list)


def add_new_data():
    for key in expression_dict.keys():
        ReRangeData(key)

def add_Train_data():
    for key in expression_dict.keys():
        ReRangeTrainData(key)

def add_Validate_data():
    for key in expression_dict.keys():
        ReRangeValidateData(key)

def add_Test_data():
    for key in expression_dict.keys():
        ReRangeTestData(key)

def petchDataFromFile():
    expression_path = 'E:/work_dir/expression_database/expression_face'
    processAnimate(expression_path)

def prepareFER2013():
    datapath = os.path.join('data', 'Fer2013.h5')
    processFER2013()
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

def prepareImageDataSet(dirpath, datapath):
    processAnimate(dirpath)
    add_new_data()
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

def prepare2ImageDataSet(dirpath, dirpaht1, datapath):
    processAnimate(dirpath)
    processAnimate(dirpaht1)
    add_new_data()
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

def clearExpList():
    for key in expression_dict.keys():
        expression_list[key].clear()

def clearList():
    global Training_x
    global Training_y
    global PublicTest_x
    global PublicTest_y
    global PrivateTest_x
    global PrivateTest_y
    global expression_list

    Training_x.clear()
    Training_y.clear()
    PublicTest_x.clear()
    PublicTest_y.clear()
    PrivateTest_x.clear()
    PrivateTest_y.clear()
    for key in expression_dict.keys():
        expression_list[key].clear()

def prepareSimpleDatasets():
    # prepareFER2013()
    #clearList()
    datapath = os.path.join('data', 'data_CKJAFFED.h5')
    CK_path = 'E:/work_dir/expression_database/expression_face/ck_classifiy'
    jaffed_path = 'E:/work_dir/expression_database/expression_face/jaffe_face'
    prepare2ImageDataSet(CK_path, jaffed_path, datapath)
    clearList()

    datapath = os.path.join('data', 'data_CK.h5')
    prepareImageDataSet(CK_path, datapath)
    clearList()

    datapath = os.path.join('data', 'data_animate.h5')
    animate_path = 'E:/work_dir/expression_database/expression_face/FERG_DB_256/FERG_DB_256'
    prepareImageDataSet(animate_path, datapath)
    clearList()

def processFER2013Image():
    train_path = 'E:/work_dir/expression_database/expression_face/FER2013/'
    validate_path = 'E:/work_dir/expression_database/Validate_FER2013/'
    test_path = 'E:/work_dir/expression_database/Test_FER2013/'
    processAnimate(train_path)
    add_Train_data()
    clearExpList()
    processAnimate(validate_path)
    add_Validate_data()
    clearExpList()
    processAnimate(test_path)
    add_Test_data()
    clearExpList()

def GenSimpleDatasets():
    # print('Exp Wild')
    # wild_path = 'E:/work_dir/expression_database/expression_face/ExpW'
    # wild_cfile = os.path.join('data', 'ExpW.h5')
    # generateSimpleDatasets(wild_path, wild_cfile)
    print('jaffe_face')
    Jaffed_path = 'E:/work_dir/expression_database/expression_face/jaffe_face'
    Jaffed_cfile = os.path.join('data', 'Jaffe.h5')
    generateSimpleDatasets(Jaffed_path, Jaffed_cfile)
    # print('CK+')
    # CK_path = 'E:/work_dir/expression_database/expression_face/ck_classifiy'
    # ck_cfile = os.path.join('data', 'CK+_split.h5')
    # generateSimpleDatasets(CK_path, ck_cfile)
    # print('animate')
    # animate_path = 'E:/work_dir/expression_database/expression_face/FERG_DB_256/FERG_DB_256'
    # animate_cfile = os.path.join('data', 'animate_split.h5')
    # generateSimpleDatasets(animate_path, animate_cfile)

    print("Save data finish!!!")


def generateSimpleDatasets(images_path, savefile_path):
    processAnimate(images_path)
    add_new_data()
    print(np.shape(Training_x))
    print(np.shape(PublicTest_x))
    print(np.shape(PrivateTest_x))

    datafile = h5py.File(savefile_path, 'w')
    datafile.create_dataset("Training_pixel", dtype = 'uint8', data=Training_x)
    datafile.create_dataset("Training_label", dtype = 'int64', data=Training_y)
    datafile.create_dataset("PublicTest_pixel", dtype = 'uint8', data=PublicTest_x)
    datafile.create_dataset("PublicTest_label", dtype = 'int64', data=PublicTest_y)
    datafile.create_dataset("PrivateTest_pixel", dtype = 'uint8', data=PrivateTest_x)
    datafile.create_dataset("PrivateTest_label", dtype = 'int64', data=PrivateTest_y)
    datafile.close()

    print("Save data finish!!!")
    clearList()

def prepareSplitDatasets():
    global datapath
    # processFER2013()
    processFER2013Image()
    print(np.shape(Training_x))
    print(np.shape(PublicTest_x))
    print(np.shape(PrivateTest_x))

    wild_path = 'E:/work_dir/expression_database/expression_face/ExpW'
    processAnimate(wild_path)
    add_new_data()
    print(np.shape(Training_x))
    print(np.shape(PublicTest_x))
    print(np.shape(PrivateTest_x))
    clearExpList()
    Jaffed_path = 'E:/work_dir/expression_database/expression_face/jaffe_face'
    processAnimate(Jaffed_path)
    add_new_data()
    print(np.shape(Training_x))
    print(np.shape(PublicTest_x))
    print(np.shape(PrivateTest_x))
    clearExpList()
    CK_path = 'E:/work_dir/expression_database/expression_face/ck_classifiy'
    processAnimate(CK_path)
    add_new_data()
    print(np.shape(Training_x))
    print(np.shape(PublicTest_x))
    print(np.shape(PrivateTest_x))
    clearExpList()
    animate_path = 'E:/work_dir/expression_database/expression_face/FERG_DB_256/FERG_DB_256'
    processAnimate(animate_path)
    add_new_data()
    print(np.shape(Training_x))
    print(np.shape(PublicTest_x))
    print(np.shape(PrivateTest_x))
    clearExpList()

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

if __name__ == '__main__':
    #GenSimpleDatasets()
    # processFER2013ToImage(bast_path, flag)
    prepareSplitDatasets()
    #prepareSimpleDatasets()

    # datapath = os.path.join('data', 'data_mixed_128.h5')
    # processFER2013()
    # animate_path = 'E:/work_dir/expression_database/expression_face'
    # processAnimate(animate_path)
    #
    # # Jaffed_path = 'E:/work_dir/expression_database/expression_face/jaffe_face'
    # # processAnimate(Jaffed_path)
    # # CK_path = 'E:/work_dir/expression_database/expression_face/ck_classifiy'
    # # processAnimate(CK_path)
    # add_new_data()
    #
    # processFER2013()
    # print(np.shape(Training_x))
    # print(np.shape(PublicTest_x))
    # print(np.shape(PrivateTest_x))
    # #
    # datafile = h5py.File(datapath, 'w')
    # datafile.create_dataset("Training_pixel", dtype = 'uint8', data=Training_x)
    # datafile.create_dataset("Training_label", dtype = 'int64', data=Training_y)
    # datafile.create_dataset("PublicTest_pixel", dtype = 'uint8', data=PublicTest_x)
    # datafile.create_dataset("PublicTest_label", dtype = 'int64', data=PublicTest_y)
    # datafile.create_dataset("PrivateTest_pixel", dtype = 'uint8', data=PrivateTest_x)
    # datafile.create_dataset("PrivateTest_label", dtype = 'int64', data=PrivateTest_y)
    # datafile.close()
    # #
    # print("Save data finish!!!")
import numpy as np
import math
import cv2

import os
import csv
from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img

impad = 4
subpath_origin = '0'
i = 1
origin_path = 'E:/Work_dir/tensorflow_expression/Facial-Expression-Recognition/data/fer2013/datasets/train/'
dst_path = 'E:/Work_dir/tensorflow_expression/Facial-Expression-Recognition/data/Augmentpre_train/'
testorigin_path = 'E:/Work_dir/tensorflow_expression/Facial-Expression-Recognition/data/test_only/'
testdst_path = 'E:/Work_dir/tensorflow_expression/Facial-Expression-Recognition/data/Augmentpre_test/'

img_row = 48
img_col = 48
csv_dataw = []

dategen = ImageDataGenerator(
    rotation_range = 4,
    shear_range  = 0.08,
    zoom_range = 0.08,
    horizontal_flip = True,
    fill_mode = 'nearest')

def read_img(path, dst_path, resize_row,resize_col):
    img_lists = []
    sc = 0
    global subpath_origin
    for filename in os.listdir(path):
        img = cv2.imread(path + '/' + filename)
        img_lists.append(img)
        sc += 1
    return img_lists, sc

def OpencvAugementProcess(img_enhanced, resize_row, resize_col):
  width = img_enhanced.shape[0]
  height = img_enhanced.shape[1]
  image_augment_lists = []
  image_temp1 = img_enhanced[0:height-impad, 0:width-impad]
  image_resize1 =  cv2.resize(image_temp1, (resize_col, resize_row))
  if(i != 3):
      image_augment_lists.append(image_resize1)
  image_temp2 = img_enhanced[impad:height, 0:width-impad]
  image_resize2 = cv2.resize(image_temp2, (resize_col, resize_row))
  if (i != 3):
      image_augment_lists.append(image_resize2)
  image_temp3 = img_enhanced[impad:height, impad:width]
  image_resize3 = cv2.resize(image_temp3, (resize_col, resize_row))
  image_augment_lists.append(image_resize3)
  image_temp4 = img_enhanced[0:height-impad, impad:width]
  image_resize4 = cv2.resize(image_temp4, (resize_col, resize_row))
  image_augment_lists.append(image_resize4)
  cpad = int(impad/2)
  image_temp5 = img_enhanced[cpad:height-cpad, cpad:width-cpad]
  image_resize5 = cv2.resize(image_temp5, (resize_col, resize_row))
  image_augment_lists.append(image_resize5)
  return image_augment_lists

def AugmentProcess(path, dst_path, Augflag=False):
    image_lists, augstart_idx = read_img(path, dst_path, img_row, img_col)
    if Augflag==True:
        fauglist = []
        if i == 1:
            fauglist = []

        for img_temp in image_lists:
            if i == 1:
                fauglist.append(img_temp)
            # auglist = OpencvAugementProcess(img_temp,  img_row, img_col)
            # for aug_elem in auglist:
            #     dst_filename = dst_path  + 'Aug_clip_' + str(augstart_idx) + '.jpg'
            #     cv2.imwrite(dst_filename, aug_elem)
            #     augstart_idx += 1
            #     if i == 1:
            #         fauglist.append(aug_elem)
        if i == 1:
            d_array = np.asarray(fauglist)
            KerasAugmentProcessBatch(d_array, dst_path, 16)

def KerasAugmentProcessBatch(data_array, dst_path, size_of_batch):
  j = 0
  seed = 1
  try:
      for batch in dategen.flow(
                data_array,
                batch_size=size_of_batch,
                seed = seed,
                save_to_dir=dst_path,
                save_prefix='Aug_',
                save_format='jpg'):
         j+= 1
         if j > 450:
            break
  except ValueError:
      print("Oops!  That was no valid number.  Try again...")


def PrepareImageData(path, dst_path, Augflag=False):
  global i
  global subpath_origin
  if not os.path.isdir(path) and not os.path.isfile(path):
      return False
  if os.path.isfile(path):
      file_path = os.path.split(path)
      lists = file_path[1].split('.')
      file_ext = lists[-1]
      img_ext = ['bmp', 'jpeg', 'gif', 'psd', 'png', 'jpg']
  elif os.path.isdir(path):
      for x in os.listdir(path):
          cur_string = os.path.join(path, x)
          if os.path.isdir(cur_string):
              subpath_origin = x
              AugmentProcess(cur_string, dst_path, Augflag)

def WriteCSVTraining(path, usage_str):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data_csv = np.asarray(img).reshape(1,48 * 48)
    data_test = data_csv.tolist()
    str_data = []
    for dn in data_test[0]:
        str_temp = str(dn)
        str_data.append(str_temp)
    s = " "
    s_new = s.join(str_data)
    tuple_new = (str(subpath_origin), s_new, usage_str)
    return tuple_new

def GenerateCSVData(path, usage_str):
  global csv_dataw
  global i
  global subpath_origin
  if not os.path.isdir(path) and not os.path.isfile(path):
      return False
  if os.path.isfile(path):
      file_path = os.path.split(path)
      lists = file_path[1].split('.')
      file_ext = lists[-1]
      img_ext = ['bmp', 'jpeg', 'gif', 'psd', 'png', 'jpg']
      if file_ext in img_ext:
          temp_tup = WriteCSVTraining(path, usage_str)
          csv_dataw.append(temp_tup)
          i += 1
  elif os.path.isdir(path):
      for x in os.listdir(path):
          cur_string = os.path.join(path, x)
          if os.path.isdir(cur_string):
              subpath_origin = x
          GenerateCSVData(cur_string, usage_str)

def BuildNewCsvFile(path, augpath):
    csvname = path
    csvfile = open(csvname, 'w', newline='')  # file('e:/csv_test.csv', 'wb')
    writer = csv.writer(csvfile)
    writer.writerow(['emotion', 'pixels', 'Usage'])
    GenerateCSVData(augpath, 'privateTest')
    writer.writerows(csv_dataw)
    csvfile.close()

def GenerateEachExpressionFile(str):
    if str == '0':
        filename = 'D:/Pre_AngryCSVFile.csv'
    elif str == '1':
        filename = 'D:/Pre_DisgustCSVFile.csv'
    elif str == '2':
        filename = 'D:/Pre_FearCSVFile.csv'
    elif str == '3':
        filename = 'D:/Pre_HappyCSVFile.csv'
    elif str == '4':
        filename = 'D:/Pre_SadCSVFile.csv'
    elif str == '5':
        filename = 'D:/Pre_SurpriseCSVFile.csv'
    elif str == '6':
        filename = 'D:/Pre_NeutralCSVFile.csv'
    BuildNewCsvFile(filename,
                    'E:/Work_dir/tensorflow_expression/Facial-Expression-Recognition/data/Augmentpre_test/'+str)

def CsvFileAdd(path):
    csvname = path
    csvfile = open(csvname, 'a+', newline='')
    writer = csv.writer(csvfile)
    #vallidpath = 'E:/Work_dir/tensorflow_expression/Facial-Expression-Recognition/data/Augmentpre_valid/'
    testpath = 'E:/Work_dir/tensorflow_expression/Facial-Expression-Recognition/data/Augmentpre_test/'
    #GenerateCSVData(vallidpath, 'PublicTest')
    GenerateCSVData(testpath, 'PrivateTest')
    writer.writerows(csv_dataw)
    csvfile.close()

def CsvFileMerge(src1csvname, src2csvname, dstcsvname):
    merged_result = []
    csvfile1 = open(src1csvname, "r")
    reader = csv.reader(csvfile1)
    for item in reader:
        if reader.line_num == 1:
            continue
        else:
            merged_result.append(item)
    csvfile1.close()
    csvfile2 = open(src2csvname, "r")
    reader2 = csv.reader(csvfile2)
    line_count = 0
    for item in reader2:
        if reader2.line_num == 1:
            continue
        else:
            merged_result.append(item)
    csvfile2.close()
    dstcsvfile = open(dstcsvname, 'a+', newline='')
    writer = csv.writer(dstcsvfile)
    writer.writerows(merged_result)
    dstcsvfile.close()

def main():
    validorigin_path = 'E:/work_dir/expression_database/expression_face/FER2013/Surprise/'
    validdst_path = 'E:/work_dir/expression_database/FER2013_aug/Surprise/'
    AugmentProcess(validorigin_path, validdst_path, True)


if __name__ == '__main__':
  main()
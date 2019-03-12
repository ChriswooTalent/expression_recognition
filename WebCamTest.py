import mxnet as mx
import numpy as np
from PIL import Image
from visualize import rgb2gray
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable
import transforms as transforms
from skimage import io
from skimage.transform import resize
from ImageEnhance import *
from models import *
import cv2
import time

CASC_PATH = 'E:/Work_dir/tensorflow_expression/Facial-Expression-Recognition/data/haarcascade_files/haarcascade_frontalface_default.xml'
LBP_PATH = 'E:/Work_dir/tensorflow_expression/Facial-Expression-Recognition/data/haarcascade_files/lbpcascade_frontalcatface.xml'
cascade_classifier = cv2.CascadeClassifier(CASC_PATH)
lbp_classifier = cv2.CascadeClassifier(LBP_PATH)
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

def format_image(image):
  if len(image.shape) > 2 and image.shape[2] == 3:
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  faces = cascade_classifier.detectMultiScale(
    image_gray,
    scaleFactor = 1.3,
    minNeighbors = 5
  )
  # None is no face found in image
  if not len(faces) > 0:
    return None, None, None
  max_are_face = faces[0]
  for face in faces:
    if face[2] * face[3] > max_are_face[2] * max_are_face[3]:
      max_are_face = face
  # face to image
  face_coor =  max_are_face
  image_gray = image_gray[face_coor[1]:(face_coor[1] + face_coor[2]), face_coor[0]:(face_coor[0] + face_coor[3])]
  roi_color = image[face_coor[1]:(face_coor[1] + face_coor[2]), face_coor[0]:(face_coor[0] + face_coor[3])]
  # Resize image to network size
  try:
    image_gray = cv2.resize(image_gray, (48, 48), interpolation=cv2.INTER_CUBIC)
    roi_color = cv2.resize(roi_color, (48, 48), interpolation=cv2.INTER_CUBIC)
  except Exception:
    print("[+} Problem during resize")
    return None, None, None
  return  image, roi_color, face_coor

def face_dect(image):
  """
  Detecting faces in image
  :param image:
  :return:  the coordinate of max face
  """
  if len(image.shape) > 2 and image.shape[2] == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  faces = cascade_classifier.detectMultiScale(
    image,
    scaleFactor = 1.3,
    minNeighbors = 5
  )
  if not len(faces) > 0:
    return None
  max_face = faces[0]
  for face in faces:
    if face[2] * face[3] > max_face[2] * max_face[3]:
      max_face = face
  face_image = image[max_face[1]:(max_face[1] + max_face[2]), max_face[0]:(max_face[0] + max_face[3])]
  try:
    image = cv2.resize(face_image, (48, 48), interpolation=cv2.INTER_CUBIC) / 255.
  except Exception:
    print("[+} Problem during resize")
    return None
  return face_image

def resize_image(image, size):
  try:
    image = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC) / 255.
  except Exception:
    print("+} Problem during resize")
    return None
  return image

def Camdemo(showBox=False):
  f = open('C:/FER_Log/Log_test.txt', 'w+')
  cut_size = 46
  space_count = 0
  transform_test = transforms.Compose([
      transforms.TenCrop(cut_size),
      transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
  ])
  net = VGG('VGG19')
  checkpoint = torch.load(os.path.join('FER2013_VGG19', 'PrivateTest_model.t7'))
  net.load_state_dict(checkpoint['net'])
  net.cuda()
  net.eval()

  feelings_faces = []
  for index, emotion in enumerate(EMOTIONS):
    feelings_faces.append(cv2.imread('E:/Work_dir/tensorflow_expression/Facial-Expression-Recognition/data/emojis/' + emotion + '.png', -1))
  video_captor = cv2.VideoCapture(0)

  emoji_face = []
  result = None
  space_count = 0
  old_space_count = 0
  while True:
    ret, frame = video_captor.read()
    detected_face, detected_face_c3, face_coor = format_image(frame)
    if showBox:
      if face_coor is not None:
        [x,y,w,h] = face_coor
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    if cv2.waitKey(1) & 0xFF == ord(' '):
      space_count = space_count+1
      if detected_face is not None:
        #cv2.imwrite('d:/detected_face.jpg', detected_face_c3)
        img_brightness = LuEnhance(detected_face_c3)
        #cv2.imwrite('d:/enhanced_face.jpg', img_brightness)
        gray = rgb2gray(img_brightness)
        gray = resize(gray, (48, 48), mode='symmetric').astype(np.uint8)

        img = gray[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        inputs = transform_test(img)
        ncrops, c, h, w = np.shape(inputs)

        inputs = inputs.view(-1, c, h, w)
        inputs = inputs.cuda()
        inputs = Variable(inputs, volatile=True)
        outputs = net(inputs)

        outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

        score = F.softmax(outputs_avg)
        _, predicted = torch.max(outputs_avg.data, 0)
        result = int(predicted.cpu().numpy())
    for index, emotion in enumerate(EMOTIONS):
      #cv2.putText(frame, emotion, (10, index * 20 + 20), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)
      if(result != None):
        #cv2.rectangle(frame, (130, index * 20 + 10), (130 + int(result * 100), (index + 1) * 20 + 4),
        #              (255, 0, 0), -1)
        emoji_face = feelings_faces[result]
        if old_space_count != space_count:
            cur_time_str = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
            emoji_str = ''
            if result == 0:
                emoji_str = 'angry'
            elif result == 1:
                emoji_str = 'disgust'
            elif result == 2:
                emoji_str = 'fear'
            elif result == 3:
                emoji_str = 'happy'
            elif result == 4:
                emoji_str = 'sad'
            elif result == 5:
                emoji_str = 'surprise'
            elif result == 6:
                emoji_str = 'Neutral'
            final_str = cur_time_str + ':current expressioni is:'+emoji_str
            print(final_str)
            f.write(final_str+'\n')
            old_space_count = old_space_count+1

        for c in range(0, 3):
          frame[200:320, 10:130, c] = emoji_face[:, :, c] * (emoji_face[:, :, 3] / 255.0) + frame[200:320, 10:130, c] * (1.0 - emoji_face[:, :, 3] / 255.0)
      cv2.imshow('face', frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  f.close()

if __name__ == '__main__':
    Camdemo()
import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from visualize import rgb2gray
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import transforms as transforms
from skimage import io
from skimage.transform import resize
from ImageEnhance import *
import cv2
import time
import argparse
from mobilenetv2 import mobilenetv2
from mobilenet_v1 import mobilenet, MobileNet, mobilenet_05
from mobileResNet_v1 import  mobileResnet, MobileResNet


parser = argparse.ArgumentParser(description='PyTorch Fer2013 CNN Training')
parser.add_argument('--model', type=str, default='mobileNet_V1', help='CNN architecture')
opt = parser.parse_args()
use_cuda = torch.cuda.is_available()
# Model
if opt.model == 'mobileNet_V1':
    net = mobilenet(num_classes=7)
elif opt.model == 'mobileNet_05':
    net = mobilenet_05(num_classes=7)
elif opt.model == 'mobileResNet_v1':
    net = mobileResnet(num_classes=7)
elif opt.model == 'mobilenetv2':
    net = mobilenetv2(num_classes=7, input_size=96)

checkpoint = torch.load(os.path.join('FER2013_mobileNet_V1/model_acc88.30', 'PrivateTest_model.t7'))
net.load_state_dict(checkpoint['net'])
if use_cuda:
    net = net.cuda()
cut_size = 90

CASC_PATH = 'E:/opencv/build/etc/haarcascades/haarcascade_frontalface_default.xml'
LBP_PATH = 'E:/opencv/build/etc/lbpcascadeslbpcascade_frontalcatface.xml'
cascade_classifier = cv2.CascadeClassifier(CASC_PATH)
lbp_classifier = cv2.CascadeClassifier(LBP_PATH)
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def plotResult(predicted, score):
    plt.clf()  # 清空画布上的所有内容
    axes=plt.subplot(1, 2, 1)

    plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.9, hspace=0.02, wspace=0.3)
    plt.subplot(1, 2, 1)
    ind = 0.1+0.6*np.arange(len(class_names))    # the x locations for the groups
    width = 0.4       # the width of the bars: can also be len(x) sequence
    color_list = ['red','orangered','darkorange','limegreen','darkgreen','royalblue','navy']
    for i in range(len(class_names)):
        plt.bar(ind[i], score[i], width, color=color_list[i])
    plt.title("Classification results ", fontsize=8)
    plt.xlabel(" Expression Category ", fontsize=8)
    plt.ylabel(" Classification Score ", fontsize=10)
    plt.xticks(ind, class_names, rotation=45, fontsize=10)

    axes=plt.subplot(1, 2, 2)
    emojis_img = io.imread('images/emojis/%s.png' % str(class_names[int(predicted.cpu().numpy())]))
    plt.imshow(emojis_img)
    plt.xlabel('Emoji Expression', fontsize=16)
    axes.set_xticks([])
    axes.set_yticks([])
    plt.tight_layout()
    # show emojis
    plt.savefig(os.path.join('images/results/2.png'))
    plt.close()
    result_img = cv2.imread('images/results/2.png')
    cv2.imshow('result', result_img)

def format_image(image):
    if len(image.shape) > 2 and image.shape[2] == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade_classifier.detectMultiScale(image_gray, scaleFactor=1.3, minNeighbors=5)
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
        image_gray = cv2.resize(image_gray, (100, 100), interpolation=cv2.INTER_CUBIC)
        roi_color = cv2.resize(roi_color, (100, 100), interpolation=cv2.INTER_CUBIC)
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
        image = cv2.resize(face_image, (100, 100), interpolation=cv2.INTER_CUBIC) / 255.
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

transform_test = transforms.Compose([
    transforms.CenterCrop(cut_size),
    transforms.ToTensor(),
])

def Camdemo():
    global cut_size
    space_count = 0
    net.eval()
    video_captor = cv2.VideoCapture(0)
    while True:
        ret, frame = video_captor.read()
        detected_face, detected_face_c3, face_coor = format_image(frame)
        if detected_face is not None:
            img_brightness = LuEnhance(detected_face_c3)
            gray = rgb2gray(img_brightness)
            gray = resize(gray, (100, 100), mode='symmetric').astype(np.uint8)
            img = gray[:, :, np.newaxis]
            img = np.concatenate((img, img, img), axis=2)
            img = Image.fromarray(img)
            inputs = transform_test(img)
            c, h, w = np.shape(inputs)
            inputs = inputs.view(-1, c, h, w)
            inputs = inputs.cuda()
            inputs = Variable(inputs, volatile=True)
            outputs = net(inputs)
            score = F.softmax(outputs)
            socre_cpu = score.data.cpu().numpy()
            _, predicted = torch.max(outputs.data[0], 0)
            plotResult(predicted, socre_cpu[0])
        cv2.imshow('cam', frame)
        cv2.waitKey(20)

if __name__ == '__main__':
    plt.rcParams['figure.figsize'] = (4.8, 3.2)
    Camdemo()
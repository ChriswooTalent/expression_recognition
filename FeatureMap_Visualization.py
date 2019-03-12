import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from skimage import io
from skimage.transform import resize
import os
import argparse
from fer import FER2013

from torch.autograd import Variable
import torchvision
import transforms as transforms
from sklearn.metrics import confusion_matrix
from mobilenet_v1 import mobilenet, MobileNet
from mobileResNet_v1 import  mobileResnet, MobileResNet
from models import *
import cv2
import time

images_pr = 16

def preprocess_image(cv2im, resize_im=True):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (Pytorch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    if resize_im:
        cv2im = cv2.resize(cv2im, (224, 224))
    im_as_arr = np.float32(cv2im)
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var

class FeatureVisualization():
    def __init__(self, image_data, selected_layer, trained_model):
        self.image_data = image_data
        self.selected_layer = selected_layer
        self.pretrained_model = trained_model

    def process_image(self):
        img=cv2.imread(self.img_path)
        img=preprocess_image(img)
        return img

    def get_feature(self):
        # input = Variable(torch.randn(1, 3, 224, 224))
        input = self.image_data
        print(input.shape)
        x = input
        layer_list = self.selected_layer
        list_len = len(layer_list)
        net_list = []
        for ly_ind in layer_list:
            temp_net = nn.Sequential(*list(self.pretrained_model.children())[:ly_ind])
            net_list.append(temp_net)
            temp_out = temp_net(x)
            temp_arr = temp_out.data.cpu().numpy()
            temp_arr = temp_arr.reshape((temp_arr.shape[1], temp_arr.shape[2], temp_arr.shape[3]))

            row_nums = int(int(temp_arr.shape[0]) / int(images_pr))
            if row_nums==0:
                row_nums=1
            col_nums = int(temp_arr.shape[0]) % images_pr
            if col_nums==0:
                col_nums=images_pr
            fig, axarr = plt.subplots(row_nums, col_nums, sharey=True, sharex=True)
            for idx in range(temp_arr.shape[0]):
                temp_image = temp_arr[idx, :, :]
                if row_nums > 1:
                    row_idx = int(idx/int(images_pr))
                    cols_idx = idx % int(images_pr)
                    axarr[row_idx, cols_idx].imshow(temp_image)
                else:
                    axarr[idx].imshow(temp_image)
            index_str = str(ly_ind)
            file_str = os.path.join('feature/%s.png' %(index_str))
            plt.xticks([])
            plt.yticks([])
            plt.savefig(file_str)
            plt.close()

    def get_single_feature(self):
        features = self.get_feature()
        print(features.shape)

        feature = features[:, 0, :, :]
        print(feature.shape)

        feature = feature.view(feature.shape[1], feature.shape[2])
        print(feature.shape)

        return feature

    def save_feature_to_img(self):
        #to numpy
        feature = self.get_single_feature()
        feature = feature.data.numpy()

        #use sigmod to [0,1]
        feature = 1.0/(1+np.exp(-1*feature))

        # to [0,255]
        feature = np.round(feature*255)
        print(feature[0])

        cv2.imwrite('./img.jpg', feature)


cut_size = 90
# Data
transform_test = transforms.Compose([
    transforms.CenterCrop(cut_size),
    transforms.ToTensor(),
])

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

if __name__ == '__main__':
    net = mobilenet(num_classes=7)
    CheckPoint = torch.load(os.path.join('FER2013_mobileNet_V1/model_acc88.30', 'PublicTest_model.t7'))

    #CheckPoint = torch.load(os.path.join('FER2013_mobileNet_V1/model_acc88.30', 'PrivateTest_model.t7'), map_location='cpu')
    net.load_state_dict(CheckPoint['net'])
    net.cuda()
    net.eval()
    raw_img = io.imread('images/2.jpg')
    gray = rgb2gray(raw_img)
    gray = resize(gray, (96, 96), mode='symmetric').astype(np.uint8)
    img = gray[:, :, np.newaxis]
    img = np.concatenate((img, img, img), axis=2)
    img = Image.fromarray(img)
    inputs = transform_test(img)
    # intputs = inputs.cpu()
    inputs = inputs[np.newaxis, :, :, :]

    ncrops, c, h, w = np.shape(inputs)
    inputs = inputs.view(-1, c, h, w)
    inputs = inputs.cuda()
    inputs = Variable(inputs, volatile=True)

    outputs = net(inputs)

    score = F.softmax(outputs)
    print(outputs)
    #feature_obj = FeatureVisualization(inputs, range(16, 18), net)
    #feature_obj.save_feature_to_img()
    #
    # outputs = net(inputs)

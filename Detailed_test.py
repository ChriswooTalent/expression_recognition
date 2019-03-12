"""
visualize results for test image
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable
from fer import FER2013
import utils

import transforms as transforms
from skimage import io
from skimage.transform import resize
from models import *
from mobilenet_v1 import mobilenet, MobileNet
from mobileResNet_v1 import  mobileResnet, MobileResNet

data_file = './data/data_mixed.h5'
t_length = 74925
v_length = 9366
te_length = 9369
re_length = 96
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
use_cuda = torch.cuda.is_available()
cut_size = 90
# Data
print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

transform_test1 = transforms.Compose([
    transforms.RandomCrop(cut_size),
    transforms.ToTensor(),
])

PublicTestset = FER2013(split = 'PublicTest', filename=data_file, train_length=t_length, validate_length=v_length, test_length=te_length, resize_length=re_length, transform=transform_test)
PublicTestloader = torch.utils.data.DataLoader(PublicTestset, batch_size=1, shuffle=False)
PrivateTestset = FER2013(split = 'PrivateTest', filename=data_file, train_length=t_length, validate_length=v_length, test_length=te_length, resize_length=re_length, transform=transform_test)
PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=1, shuffle=False)

net = mobilenet(num_classes=7)
checkpoint = torch.load(os.path.join('FER2013_mobileNet_V1/model_acc88.30', 'PrivateTest_model.t7'))
net.load_state_dict(checkpoint['net'])
net.cuda()

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def detailed_test_single():
    net.eval()
    raw_img = io.imread('images/2.jpg')
    gray = rgb2gray(raw_img)
    gray = resize(gray, (re_length, re_length), mode='symmetric').astype(np.uint8)
    img = gray[:, :, np.newaxis]
    img = np.concatenate((img, img, img), axis=2)
    img = Image.fromarray(img)
    inputs_cmp = transform_test(img)
    inputs = transform_test1(img)

    c, h, w = np.shape(inputs)
    ncrops = 1
    #ncrops, c, h, w = np.shape(inputs)
    inputs = inputs.view(-1, c, h, w)
    inputs = inputs.cuda()
    inputs = Variable(inputs, volatile=True)
    outputs = net(inputs)

    outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

    score = F.softmax(outputs_avg)
    _, predicted = torch.max(outputs_avg.data, 0)

    plt.rcParams['figure.figsize'] = (13.5,5.5)
    axes=plt.subplot(1, 3, 1)
    plt.imshow(raw_img)
    plt.xlabel('Input Image', fontsize=16)
    axes.set_xticks([])
    axes.set_yticks([])
    plt.tight_layout()

    plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.9, hspace=0.02, wspace=0.3)
    plt.subplot(1, 3, 2)
    ind = 0.1+0.6*np.arange(len(class_names))    # the x locations for the groups
    width = 0.4       # the width of the bars: can also be len(x) sequence
    color_list = ['red','orangered','darkorange','limegreen','darkgreen','royalblue','navy']
    for i in range(len(class_names)):
        plt.bar(ind[i], score.data.cpu().numpy()[i], width, color=color_list[i])
    plt.title("Classification results ",fontsize=20)
    plt.xlabel(" Expression Category ",fontsize=16)
    plt.ylabel(" Classification Score ",fontsize=16)
    plt.xticks(ind, class_names, rotation=45, fontsize=14)

    axes=plt.subplot(1, 3, 3)
    emojis_img = io.imread('images/emojis/%s.png' % str(class_names[int(predicted.cpu().numpy())]))
    plt.imshow(emojis_img)
    plt.xlabel('Emoji Expression', fontsize=16)
    axes.set_xticks([])
    axes.set_yticks([])
    plt.tight_layout()
    # show emojis

    #plt.show()
    plt.savefig(os.path.join('images/results/2.png'))
    plt.close()

    print("The Expression is %s" %str(class_names[int(predicted.cpu().numpy())]))

def PublicBatchTest():
    net.eval()
    PublicTest_loss = 0
    f_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(PublicTestloader):
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
        #loss = criterion(outputs_avg, targets)
        #PublicTest_loss += loss.data[0]
        _, predicted = torch.max(outputs_avg.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        #f_loss = float(PublicTest_loss) / float(batch_idx + 1)
        utils.progress_bar(batch_idx, len(PublicTestloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                           % (PublicTest_loss / (batch_idx + 1), 100.0 * float(correct) / float(total), correct, total))

    # Save checkpoint.
    PublicTest_acc = 100.0*float(correct)/float(total)
    print("PublicTest_acc: %0.4f" % PublicTest_acc)

def PrivateBatchTest():
    net.eval()
    PrivateTest_loss = 0
    f_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(PrivateTestloader):
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
        #loss = criterion(outputs_avg, targets)
        #PrivateTest_loss += loss.data[0]
        _, predicted = torch.max(outputs_avg.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        #f_loss = float(PrivateTest_loss) / float(batch_idx + 1)
        utils.progress_bar(batch_idx, len(PublicTestloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (PrivateTest_loss / (batch_idx + 1), 100.0 * float(correct) / float(total), correct, total))
    # Save checkpoint.
    PrivateTest_acc = 100.0*float(correct)/float(total)
    print("PrivateTest_acc: %0.4f" % PrivateTest_acc)

def detailed_batch_test():
    PublicBatchTest()
    PrivateBatchTest()

def CKJaffed_batch_test():
    global PublicTestset
    global PublicTestloader
    global PrivateTestset
    global PrivateTestloader
    data_file = './data/data_CKJAFFED.h5'
    t_length = 1604
    v_length = 200
    te_length = 203
    re_length = 96
    batchsize = 8
    PublicTestset = FER2013(split='PublicTest', filename=data_file, train_length=t_length, validate_length=v_length,
                            test_length=te_length, resize_length=re_length, transform=transform_test)
    PublicTestloader = torch.utils.data.DataLoader(PublicTestset, batch_size=batchsize, shuffle=False)
    PrivateTestset = FER2013(split='PrivateTest', filename=data_file, train_length=t_length, validate_length=v_length,
                             test_length=te_length, resize_length=re_length, transform=transform_test)
    PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=batchsize, shuffle=False)
    detailed_batch_test()

def Animata_batch_test():
    global PublicTestset
    global PublicTestloader
    global PrivateTestset
    global PrivateTestloader
    data_file = './data/data_animate.h5'
    t_length = 44610
    v_length = 5577
    te_length = 5579
    re_length = 96
    batchsize = 8
    PublicTestset = FER2013(split='PublicTest', filename=data_file, train_length=t_length, validate_length=v_length,
                            test_length=te_length, resize_length=re_length, transform=transform_test)
    PublicTestloader = torch.utils.data.DataLoader(PublicTestset, batch_size=batchsize, shuffle=False)
    PrivateTestset = FER2013(split='PrivateTest', filename=data_file, train_length=t_length, validate_length=v_length,
                             test_length=te_length, resize_length=re_length, transform=transform_test)
    PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=batchsize, shuffle=False)

    detailed_batch_test()

def CK_batch_test():
    global PublicTestset
    global PublicTestloader
    global PrivateTestset
    global PrivateTestloader
    data_file = './data/data_CK.h5'
    t_length = 1433
    v_length = 179
    te_length = 182
    re_length = 96
    batchsize = 8
    PublicTestset = FER2013(split='PublicTest', filename=data_file, train_length=t_length, validate_length=v_length,
                            test_length=te_length, resize_length=re_length, transform=transform_test)
    PublicTestloader = torch.utils.data.DataLoader(PublicTestset, batch_size=batchsize, shuffle=False)
    PrivateTestset = FER2013(split='PrivateTest', filename=data_file, train_length=t_length, validate_length=v_length,
                             test_length=te_length, resize_length=re_length, transform=transform_test)
    PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=batchsize, shuffle=False)

    detailed_batch_test()

def FER_batch_test():
    global PublicTestset
    global PublicTestloader
    global PrivateTestset
    global PrivateTestloader
    data_file = './data/Fer2013.h5'
    t_length = 28709
    v_length = 3589
    te_length = 3589
    re_length = 96
    batchsize = 8
    PublicTestset = FER2013(split='PublicTest', filename=data_file, train_length=t_length, validate_length=v_length,
                            test_length=te_length, resize_length=re_length, transform=transform_test)
    PublicTestloader = torch.utils.data.DataLoader(PublicTestset, batch_size=batchsize, shuffle=False)
    PrivateTestset = FER2013(split='PrivateTest', filename=data_file, train_length=t_length, validate_length=v_length,
                             test_length=te_length, resize_length=re_length, transform=transform_test)
    PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=batchsize, shuffle=False)
    detailed_batch_test()


if __name__ == '__main__':
    #FER_batch_test()
    #detailed_batch_test()
    #CKJaffed_batch_test()
    #CK_batch_test()
    #Animata_batch_test()
    detailed_test_single()
"""
plot confusion_matrix of PublicTest and PrivateTest
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import argparse
from fer import FER2013

from torch.autograd import Variable
import torchvision
import transforms as transforms
from sklearn.metrics import confusion_matrix
from mobilenetv2 import mobilenetv2
from mobilenet_v1 import mobilenet, MobileNet, mobilenet_05
from mobileResNet_v1 import  mobileResnet, MobileResNet
from models import *

modelfile_dict = {'mobileResNet_v1': 'FER2013_mobileResNet_v1', 'mobileNet_V1': 'FER2013_mobileNet_V1',
                  'mobileNet_05': 'FER2013_mobilenet_05', 'mobileNet_v2': 'FER2013_mobileNet_v2'}
parser = argparse.ArgumentParser(description='PyTorch Fer2013 CNN Training')
parser.add_argument('--model', type=str, default='mobileResNet_v1', help='CNN architecture')
parser.add_argument('--dataset', type=str, default='FER2013', help='CNN architecture')
parser.add_argument('--split', type=str, default='PrivateTest', help='split')
opt = parser.parse_args()

# data_file = './data/data_mixed.h5'
# t_length = 74920
# v_length = 9366
# te_length = 9374
# re_length = 96

# data_file = './data/data_Augmixed_split.h5'
# t_length = 99541
# v_length = 9366
# te_length = 9369
# re_length = 100

# data_file = './data/data_Augmixed_split_validate.h5'
# t_length = 99541
# v_length = 9366
# te_length = 9369
# re_length = 96

# data_file = './data/data_ExpW_split.h5'
# t_length = 88870
# v_length = 11110
# te_length = 11120
# re_length = 100

# confusion_matrix_file = '_mobile_FER2013.png'
# data_file = os.path.join('data/split_dataset', 'FER2013_split.h5')
# t_length = 28709
# v_length = 3589
# te_length = 3589
# re_length = 100

confusion_matrix_file = '_mobile_EXPW.png'
data_file = os.path.join('data/split_dataset', 'ExpW.h5')
t_length = 42659
v_length = 5333
te_length = 5335
re_length = 100

# confusion_matrix_file = '_mobile_Jaffe.png'
# data_file = os.path.join('data/split_dataset', 'Jaffe.h5')
# t_length = 168
# v_length = 21
# te_length = 24
# re_length = 100

# confusion_matrix_file = '_mobile_CK.png'
# data_file = os.path.join('data/split_dataset', 'CK+_split.h5')
# t_length = 1433
# v_length = 179
# te_length = 182
# re_length = 100

# confusion_matrix_file = '_mobile_animate.png'
# data_file = os.path.join('data/split_dataset', 'animate_split.h5')
# t_length = 44610
# v_length = 5577
# te_length = 5579
# re_length = 100

cut_size = 96

file_str = os.path.join(modelfile_dict[opt.model], 'PrivateTest_model.t7')

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")


    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.tight_layout()

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Model
if opt.model == 'VGG19':
    net = VGG('VGG19')
elif opt.model == 'Resnet18':
    net = ResNet18()
elif opt.model == 'mobileNet_V1':
    net = mobilenet(num_classes=7)
elif opt.model == 'mobileNet_05':
    net = mobilenet_05(num_classes=7)
elif opt.model == 'mobileResNet_v1':
    net = mobileResnet(num_classes=7)
elif opt.model == 'mobileNet_v2':
    net = mobilenetv2(num_classes=7, input_size=96)

path = os.path.join(opt.dataset + '_' + opt.model)
#checkpoint = torch.load(os.path.join('FER2013_mobileNet_V1/model_acc88.30', 'PrivateTest_model.t7'))

checkpoint = torch.load(file_str)
#checkpoint = torch.load(os.path.join(path, opt.split + '_model.t7'))

net.load_state_dict(checkpoint['net'])
net.cuda()
net.eval()

Testset = FER2013(split='PrivateTest', filename=data_file, train_length=t_length, validate_length=v_length, test_length=te_length, resize_length=re_length, transform=transform_test)
Testloader = torch.utils.data.DataLoader(Testset, batch_size=8, shuffle=False)

correct = 0
total = 0
all_target = []
for batch_idx, (inputs, targets) in enumerate(Testloader):
    bs, ncrops, c, h, w = np.shape(inputs)
    inputs = inputs.view(-1, c, h, w)
    inputs, targets = inputs.cuda(), targets.cuda()
    inputs, targets = Variable(inputs, volatile=True), Variable(targets)
    outputs = net(inputs)

    outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
    _, predicted = torch.max(outputs_avg.data, 1)

    total += targets.size(0)
    correct += predicted.eq(targets.data).cpu().sum()
    if batch_idx == 0:
        all_predicted = predicted
        all_targets = targets
    else:
        all_predicted = torch.cat((all_predicted, predicted),0)
        all_targets = torch.cat((all_targets, targets),0)

acc = 100.0 * float(correct) / float(total)
print("accuracy: %0.3f" % acc)

# Compute confusion matrix
matrix = confusion_matrix(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy())
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure(figsize=(10, 8))
plot_confusion_matrix(matrix, classes=class_names, normalize=True,
                      title= opt.split+' Confusion Matrix (Accuracy: %0.3f%%)' %acc)
plt.savefig(os.path.join(path, opt.split + confusion_matrix_file))
plt.close()

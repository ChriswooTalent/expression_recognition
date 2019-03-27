'''Train Fer2013 with PyTorch.'''
# 10 crop for data enhancement
from __future__ import print_function

import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import transforms as transforms
import numpy as np
import os
import argparse
from fer import FER2013
import utils
from mobilenetv2 import mobilenetv2
from mobilenet_v1 import mobilenet, MobileNet, mobilenet_05
from mobileResNet_v1 import  mobileResnet, MobileResNet
from focalloss import FocalLoss
from torch.autograd import Variable
from models import *

# data_file = './data/data_mixed.h5'
# t_length = 74925
# v_length = 9366
# te_length = 9369
# re_length = 96

# data_file = './data/data_pretrain.h5'
# t_length = 99540
# v_length = 9366
# te_length = 9369
# re_length = 100

data_file = './data/data_wild.h5'
t_length = 144747
v_length = 15017
te_length = 15023
re_length = 100
#
# t_length = 28709
# v_length = 3589
# te_length = 3589
# re_length = 48

training_loss = []
validation_loss = []
test_loss = []

training_acc = []
validation_acc = []
test_acc = []

parser = argparse.ArgumentParser(description='PyTorch Fer2013 CNN Training')
parser.add_argument('--model', type=str, default='Resnet18', help='CNN architecture')
parser.add_argument('--dataset', type=str, default='FER2013', help='CNN architecture')
parser.add_argument('--bs', default=64, type=int, help='learning rate')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', default=False, help='resume from checkpoint')
parser.add_argument('--warmup', default=5, type=int)
parser.add_argument('--milestones', default='25,35,45,150,220', type=str)
opt = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_PublicTest_acc = 0  # best PublicTest accuracy
best_PublicTest_acc_epoch = 0
best_PrivateTest_acc = 0  # best PrivateTest accuracy
best_PrivateTest_acc_epoch = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

learning_rate_decay_start = 50  # 50
learning_rate_decay_every = 5 # 5
learning_rate_decay_rate = 0.9 # 0.9

cut_size = 96
total_epoch = 100

path = os.path.join(opt.dataset + '_' + opt.model)

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(cut_size),
    transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(4, resample=PIL.Image.BILINEAR),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

# transform_test = transforms.Compose([
#      transforms.ToTensor(),
#  ])

trainset = FER2013(split = 'Training', filename=data_file, train_length=t_length, validate_length=v_length, test_length=te_length, resize_length=re_length, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.bs, shuffle=True)
PublicTestset = FER2013(split = 'PublicTest', filename=data_file, train_length=t_length, validate_length=v_length, test_length=te_length, resize_length=re_length, transform=transform_test)
PublicTestloader = torch.utils.data.DataLoader(PublicTestset, batch_size=int(opt.bs/2), shuffle=False)
PrivateTestset = FER2013(split = 'PrivateTest', filename=data_file, train_length=t_length, validate_length=v_length, test_length=te_length, resize_length=re_length, transform=transform_test)
PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=int(opt.bs/2), shuffle=False)

# Model
if opt.model == 'VGG19':
    net = VGG('VGG19')
elif opt.model  == 'Resnet18':
    net = ResNet18()
elif opt.model == 'mobileNet_V1':
    net = mobilenet(num_classes=7)
elif opt.model == 'mobileNet_05':
    net = mobilenet_05(num_classes=7)
elif opt.model == 'mobileResNet_v1':
    net = mobileResnet(num_classes=7)
elif opt.model == 'mobilenetv2':
    net = mobilenetv2(num_classes=7, input_size=96)

if opt.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(path,'PrivateTest_model.t7'))

    net.load_state_dict(checkpoint['net'])
    best_PublicTest_acc = checkpoint['best_PublicTest_acc']
    best_PrivateTest_acc = checkpoint['best_PrivateTest_acc']
    best_PrivateTest_acc_epoch = checkpoint['best_PublicTest_acc_epoch']
    best_PrivateTest_acc_epoch = checkpoint['best_PrivateTest_acc_epoch']
    start_epoch = checkpoint['best_PrivateTest_acc_epoch'] + 1
else:
    print('==> Building model..')


if use_cuda:
    net = net.cuda()

criterion = nn.CrossEntropyLoss()
#criterion = FocalLoss()
#optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)

def write_result_to_file(result_list, filename):
    result_file = open(filename, "w")
    file_count = 0
    for item in result_list:
        result_file.write("%s " % item)
        if ((file_count > 0) and (file_count % 10 == 0)):
            result_file.write('\n')
        file_count += 1
    result_file.close()

def adjust_learning_rate(optimizer, epoch, milestones=None):
    """Sets the learning rate: milestone is a list/tuple"""
    temp_milestone = []
    for pixel in milestones.split(','):
        temp_milestone.append(int(pixel))
    def to(epoch):
        if epoch <= opt.warmup:
            return 1
        elif opt.warmup < epoch <= temp_milestone[0]:
            return 0
        for i in range(1, len(milestones)):
            if temp_milestone[i - 1] < epoch <= temp_milestone[i]:
                return i
        return len(milestones)

    n = to(epoch)

    cur_lr = 0
    decayfactor = 0.98

    if n > 1:
        cur_lr = opt.lr * (0.2 ** 1)*(10**(-(n-1)))
    else:
        cur_lr = opt.lr * (0.2 ** n)

    if epoch <= opt.warmup:
        cur_lr = 0.00001
    if temp_milestone[0] < epoch <= temp_milestone[1]:
        cur_lr = 0.001*(1.0 ** (epoch-temp_milestone[0]-1))
    if temp_milestone[1] < epoch <= temp_milestone[2]:
        cur_lr = 0.0005*(decayfactor ** (epoch-temp_milestone[1]-1))
    if temp_milestone[2] < epoch <= temp_milestone[3]:
            cur_lr = 0.0001*(decayfactor ** (epoch-temp_milestone[2]-1))
    if temp_milestone[3] < epoch <= temp_milestone[4]:
            cur_lr = 0.00001
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    return cur_lr

# Training
def train_warmup(epoch):
    print('\nEpoch: %d' % epoch)
    global Train_acc
    net.train()
    train_loss = 0
    f_loss = 0.0
    correct = 0
    total = 0

    current_lr = adjust_learning_rate(optimizer, epoch, opt.milestones)
    print('learning_rate: %s' % str(current_lr))

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        utils.clip_gradient(optimizer, 0.1)
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        f_loss = float(train_loss) / float(batch_idx + 1)
        utils.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.0*float(correct)/float(total), correct, total))

    Train_acc = 100.0*float(correct)/float(total)
    training_loss.append(f_loss)
    training_acc.append(Train_acc)
    # print('Saving..')
    # state = {
    #     'net': net.state_dict() if use_cuda else net,
    # }
    # torch.save(state, 'train_model.t7')

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    global Train_acc
    net.train()
    train_loss = 0
    f_loss = 0.0
    correct = 0
    total = 0

    if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
        frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
        decay_factor = learning_rate_decay_rate ** frac
        current_lr = opt.lr * decay_factor
        utils.set_lr(optimizer, current_lr)  # set the decayed rate
    else:
        current_lr = opt.lr
    print('learning_rate: %s' % str(current_lr))

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        utils.clip_gradient(optimizer, 0.1)
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        f_loss = float(train_loss) / float(batch_idx + 1)
        utils.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.0*float(correct)/float(total), correct, total))

    Train_acc = 100.0*float(correct)/float(total)
    training_loss.append(f_loss)
    training_acc.append(Train_acc)

    # print('Saving..')
    # state = {
    #     'net': net.state_dict() if use_cuda else net,
    # }
    # torch.save(state, 'PublicTest_model.t7')
    # print('Train_acc is:')
    # print(Train_acc)
    #
    # print('Train loss is:')
    # print(train_loss)

def PublicTest(epoch):
    global PublicTest_acc
    global best_PublicTest_acc
    global best_PublicTest_acc_epoch
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
        loss = criterion(outputs_avg, targets)
        PublicTest_loss += loss.item()
        _, predicted = torch.max(outputs_avg.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        f_loss = float(PublicTest_loss) / float(batch_idx + 1)
        utils.progress_bar(batch_idx, len(PublicTestloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                           % (PublicTest_loss / (batch_idx + 1), 100.0 * float(correct) / float(total), correct, total))

    # Save checkpoint.
    PublicTest_acc = 100.0*float(correct)/float(total)
    validation_acc.append(PublicTest_acc)
    validation_loss.append(f_loss)
    if PublicTest_acc > best_PublicTest_acc:
        print('Saving..')
        print("best_PublicTest_acc: %0.3f" % PublicTest_acc)
        state = {
            'net': net.state_dict() if use_cuda else net,
            'acc': PublicTest_acc,
            'epoch': epoch,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path,'PublicTest_model.t7'))
        best_PublicTest_acc = PublicTest_acc
        best_PublicTest_acc_epoch = epoch

def PrivateTest(epoch):
    global Private_count
    global PrivateTest_acc
    global best_PrivateTest_acc
    global best_PrivateTest_acc_epoch
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
        loss = criterion(outputs_avg, targets)
        PrivateTest_loss += loss.item()
        _, predicted = torch.max(outputs_avg.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        f_loss = float(PrivateTest_loss) / float(batch_idx + 1)
        utils.progress_bar(batch_idx, len(PublicTestloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (PrivateTest_loss / (batch_idx + 1), 100.0 * float(correct) / float(total), correct, total))
    # Save checkpoint.
    PrivateTest_acc = 100.0*float(correct)/float(total)
    test_acc.append(PrivateTest_acc)
    test_loss.append(f_loss)
    if PrivateTest_acc > best_PrivateTest_acc:
        print('Saving..')
        print("best_PrivateTest_acc: %0.3f" % PrivateTest_acc)
        state = {
            'net': net.state_dict() if use_cuda else net,
	        'best_PublicTest_acc': best_PublicTest_acc,
            'best_PrivateTest_acc': PrivateTest_acc,
    	    'best_PublicTest_acc_epoch': best_PublicTest_acc_epoch,
            'best_PrivateTest_acc_epoch': epoch,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path,'PrivateTest_model.t7'))
        best_PrivateTest_acc = PrivateTest_acc
        best_PrivateTest_acc_epoch = epoch

for epoch in range(start_epoch, total_epoch):
    train(epoch)
    #train_warmup(epoch)
    PublicTest(epoch)
    PrivateTest(epoch)
    # write_result_to_file(training_loss, 'loss/train_loss.txt')
    # write_result_to_file(validation_loss, 'loss/validate_loss.txt')
    # write_result_to_file(test_loss, 'loss/test_loss.txt')
    #
    # write_result_to_file(training_acc, 'acc/train_acc.txt')
    # write_result_to_file(validation_acc, 'acc/validate_acc.txt')
    # write_result_to_file(test_acc, 'acc/test_acc.txt')


write_result_to_file(training_loss, 'loss/train_loss.txt')
write_result_to_file(validation_loss, 'loss/validate_loss.txt')
write_result_to_file(test_loss, 'loss/test_loss.txt')

write_result_to_file(training_acc, 'loss/train_acc.txt')
write_result_to_file(validation_acc, 'loss/validate_acc.txt')
write_result_to_file(test_acc, 'loss/test_acc.txt')


print("best_PublicTest_acc: %0.3f" % best_PublicTest_acc)
print("best_PublicTest_acc_epoch: %d" % best_PublicTest_acc_epoch)
print("best_PrivateTest_acc: %0.3f" % best_PrivateTest_acc)
print("best_PrivateTest_acc_epoch: %d" % best_PrivateTest_acc_epoch)

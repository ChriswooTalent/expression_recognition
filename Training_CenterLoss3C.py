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
from mixed import Mixed
import utils
from mobilenetv1_centerloss import mobilenetv1CL
from mobileResNet_CenterLoss import mobileResnetCL
from mobilenetv2 import mobilenetv2
from mobilenetv2_CenterLoss import mobilenetv2CL
from mobilenet_v1 import mobilenet, MobileNet, mobilenet_05
from mobileResNet_v1 import  mobileResnet, MobileResNet
from mobileDenseNet_CenterLoss import mobileDensenetCL
from mobileDenseNetV2_CenterLoss import mobileDensenetv2CL
from focalloss import FocalLoss
from torch.autograd import Variable
from models import *
from CenterLoss import CenterLoss

data_file = './data/data_wild_3C.h5'
t_length = 144747
v_length = 15017
te_length = 15023
re_length = 70

parser = argparse.ArgumentParser(description='PyTorch Fer2013 CNN Training')
parser.add_argument('--model', type=str, default='mobileDensev2_CL', help='CNN architecture')
parser.add_argument('--dataset', type=str, default='FER2013', help='CNN architecture')
parser.add_argument('--bs', default=32, type=int, help='learning rate')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', default=False, help='resume from checkpoint')
parser.add_argument('--weight_init', '-i', default=False, help='init from checkpoint')
parser.add_argument('--warmup', default=0, type=int)
parser.add_argument('--milestones', default='25,35,45,150,220', type=str)
opt = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_PublicTest_acc = 0  # best PublicTest accuracy
best_PublicTest_acc_epoch = 0
best_PrivateTest_acc = 0  # best PrivateTest accuracy
best_PrivateTest_acc_epoch = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

learning_rate_decay_start = 20  # 50
learning_rate_decay_every = 5 # 5
learning_rate_decay_rate = 0.9 # 0.9

cut_size = 64
total_epoch = 200


weight_init_path = os.path.join('FER2013_mobilev1_CL')
assert os.path.isdir(weight_init_path), 'Error: no checkpoint directory found!'
path = os.path.join(opt.dataset + '_' + opt.model)

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(cut_size),
    transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(3, resample=PIL.Image.BILINEAR),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.FiveCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

# transform_test = transforms.Compose([
#      transforms.ToTensor(),
#  ])

trainset = Mixed(split = 'Training', filename=data_file, train_length=t_length, validate_length=v_length, test_length=te_length, resize_length=re_length, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.bs, shuffle=True)
PublicTestset = Mixed(split = 'PublicTest', filename=data_file, train_length=t_length, validate_length=v_length, test_length=te_length, resize_length=re_length, transform=transform_test)
PublicTestloader = torch.utils.data.DataLoader(PublicTestset, batch_size=16, shuffle=False)
PrivateTestset = Mixed(split = 'PrivateTest', filename=data_file, train_length=t_length, validate_length=v_length, test_length=te_length, resize_length=re_length, transform=transform_test)
PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=16, shuffle=False)

cl_featurenum = 1920
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
elif opt.model == 'centerloss':
    net = mobileResnetCL(num_classes=7)
    pretrained_net = mobileResnet(num_classes=7)
elif opt.model == 'mobilev1_CL':
    net = mobilenetv1CL(num_classes=7)
    cl_featurenum = 1024
    pretrained_net = mobilenet(num_classes=7)
elif opt.model == 'mobilenetv2_CL':
    net = mobilenetv2CL(num_classes=7, input_size=96)
    pretrained_net = mobilenetv2(num_classes=1000, input_size=96)
    cl_featurenum = 1280
elif opt.model == 'mobileDensev1_CL':
    net = mobileDensenetCL(num_classes=7)
    pretrained_net = mobilenetv1CL(num_classes=7)
    cl_featurenum = 512
elif opt.model == 'mobileDensev2_CL':
    net = mobileDensenetv2CL(num_classes=7)
    pretrained_net = mobilenetv1CL(num_classes=7)
    cl_featurenum = 1920

model_dict = net.state_dict()

if opt.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(weight_init_path), 'Error: no checkpoint directory found!'
    file_path = os.path.join(weight_init_path,'PrivateTest_model.t7')
    checkpoint = torch.load(file_path)

    net.load_state_dict(checkpoint['net'])

    best_PublicTest_acc = checkpoint['best_PublicTest_acc']
    best_PrivateTest_acc = checkpoint['best_PrivateTest_acc']
    best_PrivateTest_acc_epoch = checkpoint['best_PublicTest_acc_epoch']
    best_PrivateTest_acc_epoch = checkpoint['best_PrivateTest_acc_epoch']
    start_epoch = checkpoint['best_PrivateTest_acc_epoch'] + 1
else:
    print('==> Building model..')

if opt.weight_init:
    # Load checkpoint.
    print('==> Init weight from checkpoint..')
    assert os.path.isdir(weight_init_path), 'Error: no checkpoint directory found!'
    file_path = os.path.join(weight_init_path,'PrivateTest_model.t7')
    checkpoint = torch.load(file_path)

    pretrained_net.load_state_dict(checkpoint['net'])
    pretrained_dict = pretrained_net.state_dict()

    pretrained_dict = {k:v for k, v in pretrained_dict.items() if k in model_dict}

    model_dict.update(pretrained_dict)

    for key in model_dict:
        if 'num_batches_tracked' in key:
            continue
        if 'merge' in key:
            model_dict[key].requires_grad = True
            continue
        if 'ip' in key:
            model_dict[key].requires_grad = True
            continue
        if 'Dense' in key:
            model_dict[key].requires_grad = True
            continue
        model_dict[key].requires_grad = False

    net.load_state_dict(model_dict)
    print('finished!')

else:
    print('==> Building model..')

torch.cuda.set_device(0)
if use_cuda:
    net = net.cuda(0)

criterion = nn.CrossEntropyLoss()
#criterion = FocalLoss()
#optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=opt.lr, momentum=0.9, weight_decay=5e-4)


# NLLLoss
nllloss = nn.NLLLoss().cuda() #CrossEntropyLoss = log_softmax + NLLLoss
# CenterLoss
loss_weight = 1.0
centerloss = CenterLoss(7, cl_featurenum).cuda()

# optimzer4center
optimzer4center = optim.SGD(centerloss.parameters(), lr=0.05)

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
        cur_lr = 0.000001
    if temp_milestone[0] < epoch <= temp_milestone[1]:
        cur_lr = 0.0005*(1.0 ** (epoch-temp_milestone[0]-1))
    if temp_milestone[1] < epoch <= temp_milestone[2]:
        cur_lr = 0.0001*(decayfactor ** (epoch-temp_milestone[1]-1))
    if temp_milestone[2] < epoch <= temp_milestone[3]:
            cur_lr = 0.00005*(decayfactor ** (epoch-temp_milestone[2]-1))
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
        optimzer4center.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        ip1, outputs = net(inputs)
        loss = nllloss(outputs, targets) + loss_weight * centerloss(targets, ip1)

        loss.backward()
        #utils.clip_gradient(optimizer, 0.1)

        optimizer.step()
        optimzer4center.step()

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
        optimzer4center.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        ip1, outputs = net(inputs)
        loss = nllloss(outputs, targets) + loss_weight * centerloss(targets, ip1)
        loss.backward()
        utils.clip_gradient(optimizer, 0.1)
        optimizer.step()
        optimzer4center.step()

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
        ip1, outputs = net(inputs)
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
        ip1_avg = ip1.view(bs, ncrops, -1).mean(1)
        loss = nllloss(outputs_avg, targets) + loss_weight * centerloss(targets, ip1_avg)
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
        ip1, outputs = net(inputs)
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
        ip1_avg = ip1.view(bs, ncrops, -1).mean(1)
        loss = nllloss(outputs_avg, targets) + loss_weight * centerloss(targets, ip1_avg)
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

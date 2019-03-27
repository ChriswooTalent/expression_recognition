import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import transforms as transforms

#tsne
from sklearn.manifold import TSNE

from PIL import Image
from skimage import io
from skimage.transform import resize
import os
import argparse
from fer import FER2013

from mobilenetv1_centerloss import mobilenetv1CL
from mobileResNet_CenterLoss import mobileResnetCL
from mobilenetv2 import mobilenetv2
from mobilenet_v1 import mobilenet, MobileNet, mobilenet_05
from mobileResNet_v1 import  mobileResnet, MobileResNet
from mobileDenseNet_CenterLoss import mobileDensenetCL
import cv2
import time

modelfile_dict = {'mobileResNet_v1': 'FER2013_mobileResNet_v1', 'mobileNet_V1': 'FER2013_mobileNet_V1/model_acc88.30',
                  'mobileNet_05': 'FER2013_mobilenet_05', 'mobileNet_v2': 'FER2013_mobileNet_v2',
                  'centerloss':'FER2013_centerloss', 'mobilev1_CL':'FER2013_mobilev1_CL',
                  'mobileDensev1_CL': 'FER2013_mobileDensev1_CL'}
parser = argparse.ArgumentParser(description='PyTorch Fer2013 CNN Training')
parser.add_argument('--model', type=str, default='mobileDensev1_CL', help='CNN architecture')
parser.add_argument('--dataset', type=str, default='FER2013', help='CNN architecture')
parser.add_argument('--split', type=str, default='PrivateTest', help='split')
opt = parser.parse_args()

images_pr = 16

tsne = TSNE(n_components=2, random_state=0)

# data_file = os.path.join('data', 'data_enhanced_big.h5')
# t_length = 28709
# v_length = 3589
# te_length = 3589
# re_length = 96

# data_file = './data/data_mixed.h5'
# t_length = 74925
# v_length = 9366
# te_length = 9369
# re_length = 96

confusion_matrix_file = '_mobile_augmixed.png'
data_file = './data/data_Augmixed_split.h5'
t_length = 99541
v_length = 9366
te_length = 9369
re_length = 100


# confusion_matrix_file = '_mobile_FER2013.png'
# data_file = os.path.join('data/split_dataset', 'FER2013_split.h5')
# t_length = 53330
# v_length = 3589
# te_length = 3584
# re_length = 100

# confusion_matrix_file = '_mobile_CK.png'
# data_file = os.path.join('data/split_dataset', 'CK+_split.h5')
# t_length = 1433
# v_length = 179
# te_length = 182
# re_length = 100

# confusion_matrix_file = '_mobile_Jaffe.png'
# data_file = os.path.join('data/split_dataset', 'Jaffe.h5')
# t_length = 168
# v_length = 21
# te_length = 24
# re_length = 100

cut_size = 90

file_str = os.path.join(modelfile_dict[opt.model], 'PrivateTest_model.t7')

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

Testset = FER2013(split='PrivateTest', filename=data_file, train_length=t_length, validate_length=v_length,
                  test_length=te_length, resize_length=re_length, transform=transform_test)
Testloader = torch.utils.data.DataLoader(Testset, batch_size=1, shuffle=False)

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
cpu_count = torch.cuda.device_count()
print(cpu_count)
GPUID = "1,2"
GPUIDS = (len(GPUID)+1)//2
BATCHSIZE = 20
os.environ["CUDA_VISIBLE_DEVICES"] = GPUID

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
elif opt.model == 'mobilev1_CL':
    net = mobilenetv1CL(num_classes=7)
elif opt.model == 'mobilenetv2_CL':
    net = mobilenetv2CL(num_classes=7, input_size=96)
elif opt.model == 'mobileDensev1_CL':
    net = mobileDensenetCL(num_classes=7)

path = os.path.join(opt.dataset + '_' + opt.model)
#checkpoint = torch.load(os.path.join('FER2013_mobileNet_V1/model_acc88.30', 'PrivateTest_model.t7'))

checkpoint = torch.load(file_str)
#checkpoint = torch.load(os.path.join(path, opt.split + '_model.t7'))

net.load_state_dict(checkpoint['net'])

net = nn.DataParallel(net,  device_ids=[1,2]).cuda(1)

net.eval()


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


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def singleImageResult():
    cut_size = 90
    # Data
    transform_test = transforms.Compose([
        transforms.CenterCrop(cut_size),
        transforms.ToTensor(),
    ])
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

def tSNE_Visualization():
    plt.figure(figsize=(7,6))
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'

    correct = 0
    total = 0
    ip_loader = []
    idx_loader = []
    for batch_idx, (inputs, targets) in enumerate(Testloader):
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        inputs, targets = inputs.cuda(1), targets.cuda(1)
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        ip1, outputs = net(inputs)

        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
        ip1_avg = ip1.view(bs, ncrops, -1).mean(1)
        cp1 = ip1_avg.cpu().detach().numpy()
        ip_loader.append(cp1.reshape(-1))
        idx_loader.append(targets.cpu().detach().numpy().reshape(-1)[0])

    ip1_2d = tsne.fit_transform(ip_loader)
    label_array = range(len(class_names))

    for i, c, label in zip(label_array, colors, class_names):
        index_arr = [x for x, val in enumerate(idx_loader) if val == i]
        plt.scatter(ip1_2d[index_arr, 0], ip1_2d[index_arr, 1], c=c, label=label)
    plt.legend()
    plt.show()
    plt.savefig(os.path.join('images/classify_visual/FER2013.png'))

if __name__ == '__main__':
    tSNE_Visualization()
    #feature_obj = FeatureVisualization(inputs, range(16, 18), net)
    #feature_obj.save_feature_to_img()
    #
    # outputs = net(inputs)

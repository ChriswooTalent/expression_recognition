import mxnet as mx
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import logging
import mobilenet_v1
from mobilenet_v1 import mobilenet, MobileNet, mobilenet_05
import math
import torchvision.transforms as transforms
import cv2
from mxmobilenet import get_symbol
from collections import namedtuple
import os
from PIL import Image
from skimage import io
from skimage.transform import resize
from torch.autograd import Variable
import time
logging.getLogger().setLevel(logging.INFO)

re_length = 96
class_num = 7
shift_bits = 0
cut_size = 90
# Data
transform_test = transforms.Compose([
    transforms.CenterCrop(cut_size),
    transforms.ToTensor(),
])
def MxDepthWiseBlock(convlayername, bnlayer_name, relu_name, data_in, group_num, filter_nums, seq_filter_nums, stride=1, use_global_stats=True):
    conv_dw = mx.symbol.Convolution(name=convlayername, data=data_in, num_filter=int(filter_nums), num_group=int(group_num), pad=(1, 1), kernel=(3, 3), stride=(stride, stride),
                                             no_bias=True)
    bn_dw = mx.symbol.BatchNorm(name=bnlayer_name, data=conv_dw, use_global_stats=use_global_stats, fix_gamma=False, eps=1e-5, axis=1)
    relu_dw = mx.symbol.Activation(name=relu_name, data=bn_dw, act_type='relu')
    conv_sep_name = convlayername + '_sep'
    conv_sep = mx.symbol.Convolution(name=conv_sep_name, data=relu_dw, num_filter=int(seq_filter_nums), num_group=1, pad=(0, 0), kernel=(1, 1),
                                        stride=(1, 1), no_bias=True)
    bn_sep_layer_name = bnlayer_name + '_sep'
    conv_sep_bn = mx.symbol.BatchNorm(name=bn_sep_layer_name, data=conv_sep, use_global_stats=use_global_stats, fix_gamma=False, eps=1e-5, axis=1)
    activate_layer_name = relu_name+'_sep'
    relu_sep = mx.symbol.Activation(name=activate_layer_name, data=conv_sep_bn, act_type='relu')

    return conv_dw, bn_dw, relu_sep

def TestNet_v1(widen_factor, num_classes, prelu=False, input_channel=3):
    block = MxDepthWiseBlock
    data = mx.symbol.Variable(name='data')
    conv1 = mx.symbol.Convolution(name='conv1', data=data, num_filter=32, pad=(1, 1), kernel=(3, 3), stride=(2, 2),
                                  no_bias=False)
    return conv1

def MxNetMobileNet_v1(widen_factor, num_classes, prelu=False, input_channel=3):
    block = MxDepthWiseBlock
    data = mx.symbol.Variable(name='input_1')
    conv1 = mx.symbol.Convolution(name='conv1', data=data, num_filter=int(32 * widen_factor), num_group=1, pad=(1, 1), kernel=(3, 3), stride=(2, 2),
                                  no_bias=True)
    conv1_bn = mx.symbol.BatchNorm(name='conv1_bn', data=conv1, use_global_stats=True, fix_gamma=False, eps=0.0000000001)
    relu1 = mx.symbol.Activation(name='relu1', data=conv1_bn, act_type='relu')

    conv2_1, bn2_1, depth_result2_1 = MxDepthWiseBlock('dw2_1_conv', 'dw2_1_bn', 'dw2_1_relu', relu1, int(32 * widen_factor), int(32 * widen_factor), int(64 * widen_factor))
    conv2_2, bn2_2, depth_result2_2 = MxDepthWiseBlock('dw2_2_conv', 'dw2_2_bn', 'dw2_2_relu', depth_result2_1, int(64 * widen_factor), int(64 * widen_factor), int(128 * widen_factor), stride=2)
    conv3_1, bn3_1, depth_result3_1 = MxDepthWiseBlock('dw3_1_conv', 'dw3_1_bn', 'dw3_1_relu', depth_result2_2, int(128 * widen_factor), int(128 * widen_factor), int(128 * widen_factor))
    conv3_2, bn3_2, depth_result3_2 = MxDepthWiseBlock('dw3_2_conv', 'dw3_2_bn', 'dw3_1_relu', depth_result3_1, int(128 * widen_factor), int(128 * widen_factor), int(256 * widen_factor), stride=2)
    conv4_1, bn4_1, depth_result4_1 = MxDepthWiseBlock('dw4_1_conv', 'dw4_1_bn', 'dw4_1_relu', depth_result3_2, int(256 * widen_factor), int(256 * widen_factor), int(256 * widen_factor))
    conv4_2, bn4_2, depth_result4_2 = MxDepthWiseBlock('dw4_2_conv', 'dw4_2_bn', 'dw4_2_relu', depth_result4_1, int(256 * widen_factor), int(256 * widen_factor), int(512 * widen_factor), stride=2)
    conv5_1, bn5_1, depth_result5_1 = MxDepthWiseBlock('dw5_1_conv', 'dw5_1_bn', 'dw5_1_relu', depth_result4_2, int(512 * widen_factor), int(512 * widen_factor), int(512 * widen_factor))
    conv5_2, bn5_2, depth_result5_2 = MxDepthWiseBlock('dw5_2_conv', 'dw5_2_bn', 'dw5_2_relu', depth_result5_1, int(512 * widen_factor), int(512 * widen_factor), int(512 * widen_factor))
    conv5_3, bn5_3, depth_result5_3 = MxDepthWiseBlock('dw5_3_conv', 'dw5_3_bn', 'dw5_3_relu', depth_result5_2, int(512 * widen_factor), int(512 * widen_factor), int(512 * widen_factor))
    conv5_4, bn5_4, depth_result5_4 = MxDepthWiseBlock('dw5_4_conv', 'dw5_4_bn', 'dw5_4_relu', depth_result5_3, int(512 * widen_factor), int(512 * widen_factor), int(512 * widen_factor))
    conv5_5, bn5_5, depth_result5_5 = MxDepthWiseBlock('dw5_5_conv', 'dw5_5_bn', 'dw5_5_relu', depth_result5_4, int(512 * widen_factor), int(512 * widen_factor), int(512 * widen_factor))
    conv5_6, bn4_6, depth_result5_6 = MxDepthWiseBlock('dw5_6_conv', 'dw5_6_bn', 'dw5_6_relu', depth_result5_5, int(512 * widen_factor), int(512 * widen_factor), int(1024 * widen_factor), stride=2)
    conv6, bn6, depth_result6 = MxDepthWiseBlock('dw6_conv', 'dw6_bn', 'dw6_relu', depth_result5_6, int(1024 * widen_factor), int(1024 * widen_factor), int(1024 * widen_factor))

    avg_pool6 = mx.symbol.Pooling(name='pool6', data=depth_result6 , pooling_convention='full', global_pool=True, kernel=(1, 1), pool_type='avg')
    fc_layer = mx.symbol.FullyConnected(name='fc_layer', data=avg_pool6, num_hidden=num_classes)

    group = mx.symbol.Group([data, conv1, conv1_bn, relu1, conv2_1, bn2_1, depth_result2_1, fc_layer])

    save_group = mx.symbol.Group([data, fc_layer])

    model = mx.mod.Module(symbol=fc_layer, context=mx.gpu(0), data_names=['input_1'])

    model.bind(for_training=False, data_shapes=[('input_1', (1, 3, cut_size, cut_size))])
    return fc_layer, model

def test_mxNet():
    net = mx.sym.Variable('data')
    net = mx.sym.FullyConnected(net, name='fc1', num_hidden=64)
    net = mx.sym.Activation(net, name='relu1', act_type='relu')
    net = mx.sym.FullyConnected(net, name='fc2', num_hidden=26)
    net = mx.sym.SoftmaxOutput(net, name='softmax')
    mx.viz.plot_network(net)

def main():
    test_mxNet()

def test_mxNetBind():
    net = mx.symbol.Variable('data')
    net = mx.symbol.FullyConnected(net, name='fc1', num_hidden=64)
    net = mx.symbol.Activation(net, name='relu1', act_type='relu')
    net = mx.symbol.FullyConnected(net, name='fc2', num_hidden=10)
    net = mx.symbol.SoftmaxOutput(net, name='softmax')
    testmod = mx.mod.Module(symbol=net, context=mx.gpu(), data_names=['data'], label_names=['softmax_label'])
    img_test = cv2.imread('img_cmp_test.jpg')
    img_test_mxNet = np.float32(img_test)
    mean_test = 127.5
    std=128.0
    img_test_mxNet = (img_test_mxNet-mean_test)/std
    img_test_mxNet = img_test_mxNet[np.newaxis, :]

    x = testmod.bind(for_training=False, data_shapes=[('data', (1, re_length, re_length, 3))])
    testmod.init_params(initializer=mx.init.Xavier(magnitude=2.))
    testmod.predict(img_test_mxNet)
    print('test finished')

def save_checkpoint(epoch, module, callback):
    arg_params, aux_params= module.get_params()
    callback(epoch, module.symbol, arg_params, aux_params)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def test_loading():
    model_prefix = 'mobilenet_v1'
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, 3)
    assert sym.tojson() == net.tojson()


if __name__ == '__main__':
    model_prefix = 'mobilenet_05'
    scale_factor = 0.5
    #1. load pre-trained model
    #checkpoint_fp = os.path.join('FER2013_mobileNet_V1/model_acc88.30', 'PublicTest_model.t7')

    checkpoint_fp = os.path.join('FER2013_mobilenet_05', 'PublicTest_model.t7')
    arch = 'mobilenet_05'

    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)

    checkpoint1 = checkpoint['net']
    mobile_model = getattr(mobilenet_v1, arch)(num_classes=7)

    model_dict = mobile_model.state_dict()
    for k in checkpoint1.keys():
        model_dict[k.replace('module.', '')] = checkpoint1[k]
        mobile_model.load_state_dict(model_dict)

    # exact_list = ["conv1", "layer1", "avgpool"]
    # myexactor = mobilenet_v1(mobile_model, exact_list)
    # x = myexactor(img)

    cudnn.benchmark = True
    #torch.save(mobile_model, 'mxnet_model/mobile_model.pkl')
    model = mobile_model.cuda()
    model.eval()

    #print check pt
    print(isinstance(checkpoint['net'], dict))
    print('keys of checkpoint:')
    for i in checkpoint:
        print(i)
        #print(state_dict[i].size())
    print('')

    state_dict = checkpoint['net']
    # # # state_dict也是一个字典
    print(torch.is_tensor(state_dict['conv1.weight']))
    # # 查看某个value的size
    print(state_dict['conv1.weight'].size())

    last_layer, mxmobilenet_v1 = MxNetMobileNet_v1(scale_factor, class_num)
    executor = mxmobilenet_v1.bind(for_training=False, data_shapes=[('input_1', (1, 3, re_length, re_length))])

    arg_params = dict()
    aux_params = dict()
    arg_count = 0
    aux_count = 0
    for key in state_dict:
        if 'num_batches_tracked' in key:
            continue
        if(key == 'conv1.weight'):
            key_mx = 'conv1_weight'
            arg_params[key_mx] = mx.nd.array(state_dict[key].data, mx.gpu(0))
            arg_count += 1
            continue
        if(key == 'bn1.weight'):
            key_mx = 'conv1_bn_gamma'
            arg_params[key_mx] = mx.nd.array(state_dict[key].data, mx.gpu(0))
            arg_count += 1
            continue
        if(key == 'bn1.bias'):
            key_mx = 'conv1_bn_beta'
            arg_params[key_mx] = mx.nd.array(state_dict[key].data, mx.gpu(0))
            arg_count += 1
            continue

        if 'running' in key:
            key_mx = mxmobilenet_v1._aux_names[aux_count]
            print(key, key_mx)
            print(mxmobilenet_v1._aux_params[key_mx].shape, state_dict[key].data.shape)
            aux_params[key_mx] = mx.nd.array(state_dict[key].data, mx.gpu(0))
            aux_count += 1
            continue

        if 'bn_dw.weight' in key:
            key_temp = key[shift_bits:]
            keys = key_temp.split('.')
            key_mx = keys[0] + '_bn_gamma'
            print(key, key_mx)
            print(mxmobilenet_v1._arg_params[key_mx].shape, state_dict[key].data.shape)
            arg_params[key_mx] = mx.nd.array(state_dict[key].data, mx.gpu(0))
            arg_count += 1
            continue

        if 'bn_dw.bias' in key:
            key_temp = key[shift_bits:]
            keys = key_temp.split('.')
            key_mx = keys[0] + '_bn_beta'
            print(key, key_mx)
            print(mxmobilenet_v1._arg_params[key_mx].shape, state_dict[key].data.shape)
            arg_params[key_mx] = mx.nd.array(state_dict[key].data, mx.gpu(0))
            arg_count += 1
            continue

        if 'bn_sep.weight' in key:
            key_temp = key[shift_bits:]
            keys = key_temp.split('.')
            key_mx = keys[0] + '_bn_sep_gamma'
            print(key, key_mx)
            print(mxmobilenet_v1._arg_params[key_mx].shape, state_dict[key].data.shape)
            arg_params[key_mx] = mx.nd.array(state_dict[key].data, mx.gpu(0))
            arg_count += 1
            continue

        if 'bn_sep.bias' in key:
            key_temp = key[shift_bits:]
            keys = key_temp.split('.')
            key_mx = keys[0] + '_bn_sep_beta'
            print(key, key_mx)
            print(mxmobilenet_v1._arg_params[key_mx].shape, state_dict[key].data.shape)
            arg_params[key_mx] = mx.nd.array(state_dict[key].data, mx.gpu(0))
            arg_count += 1
            continue

        key_mx = mxmobilenet_v1._param_names[arg_count]
        print(key, key_mx)
        print(mxmobilenet_v1._arg_params[key_mx].shape, state_dict[key].data.shape)
        arg_count += 1
        arg_params[key_mx] = mx.nd.array(state_dict[key].data, mx.gpu(0))

    mxmobilenet_v1.set_params(arg_params=arg_params, aux_params=aux_params, allow_missing=False)

    raw_img = io.imread('images/1.jpg')
    gray = rgb2gray(raw_img)
    gray = resize(gray, (re_length, re_length), mode='symmetric').astype(np.uint8)
    img = gray[:, :, np.newaxis]
    img = np.concatenate((img, img, img), axis=2)
    img = Image.fromarray(img)
    inputs = transform_test(img)

    inputs = inputs[np.newaxis, :, :, :]

    ncrops, c, h, w = np.shape(inputs)
    inputs = inputs.view(-1, c, h, w)
    inputs = inputs.cuda()
    mx_input = inputs.cpu().numpy()
    inputs = Variable(inputs, volatile=True)

    input_saved = mx_input
    input_saved.flatten()
    #np.savetxt("input_image.txt", input_saved)
    Batch = namedtuple('Batch', ['data'])
    mxmobilenet_v1.forward(Batch([mx.nd.array(mx_input, mx.gpu(0))]))  # 预测结果

    mx_par = mxmobilenet_v1.get_params()
    conv1_w = mxmobilenet_v1.get_params()[0]['conv1_weight']
    dw2_1_bn_w = mxmobilenet_v1.get_params()[0]['dw2_1_bn_gamma']
    dw2_1_bn_b = mxmobilenet_v1.get_params()[0]['dw2_1_bn_beta']
    dw2_1_mean = mxmobilenet_v1.get_params()[1]['dw2_1_bn_moving_mean']
    dw2_1_var = mxmobilenet_v1.get_params()[1]['dw2_1_bn_moving_var']
    outputs = mxmobilenet_v1.get_outputs()
    prob = mxmobilenet_v1.get_outputs()[0].asnumpy()
    result_cmp = mxmobilenet_v1.predict(mx.nd.array(mx_input, mx.gpu(0)))

    # construct a callback function to save checkpoints

    checkpoint = mx.callback.do_checkpoint(model_prefix)
    save_checkpoint(1000, mxmobilenet_v1, checkpoint)

    net = mobilenet_05(num_classes=7)
    CheckPoint_cmp = torch.load(checkpoint_fp)
    net.load_state_dict(CheckPoint_cmp['net'])
    net.cuda()
    net.eval()
    torch_cmp = net(inputs)
    print(torch_cmp)

    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, 1001)
    print(sym.name)
    ctx = mx.cpu(0)
    mod_new = mx.mod.Module(symbol=sym, context=ctx, data_names=['input_1'])
    mod_new.bind(for_training=False, data_shapes=[('input_1', (1, 3, cut_size, cut_size))])
    mod_new.set_params(arg_params, aux_params)
    mod_new.forward(Batch([mx.nd.array(mx_input, ctx)]))  # 预测结果
    t1 = time.time()
    for i in range(1000):
        mod_new.forward(Batch([mx.nd.array(mx_input, ctx)]))  # 预测结果
    t2 = time.time()
    outputs1 = mod_new.get_outputs()
    print(t2-t1)
    print(outputs1)
    print('finish')


    #
    # img_test1 = cv2.imread('E:/test_model/test_image.jpg')
    # imag_dst = cv2.resize(img_test1, (60, 60))
    # transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])
    # input1 = transform(imag_dst).unsqueeze(0)
    # input1_cmp = transform(imag_dst).unsqueeze(0)
    # with torch.no_grad():
    #     # if args.mode == 'gpu':
    #     input1 = input1.cuda()
    #     testmx_input = input1.cpu().numpy()
    #
    # model_prefix = 'E:/test_model/model_not_trained'
    #
    # sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, 0000)
    # print(sym.name)
    # ctx = mx.cpu(0)
    # mod_new = mx.mod.Module(symbol=sym, context=ctx, data_names=['data'])
    # mod_new.bind(for_training=False, data_shapes=[('data', (1, 3, 60, 60))])
    # mod_new.set_params(arg_params, aux_params)
    # mod_new.forward(Batch([mx.nd.array(testmx_input, ctx)]))  # 预测结果
    # t1 = time.time()
    # for i in range(1000):
    #     mod_new.forward(Batch([mx.nd.array(mx_input, ctx)]))  # 预测结果
    # t2 = time.time()
    # outputs_test = mod_new.get_outputs()
    # print(t2-t1)

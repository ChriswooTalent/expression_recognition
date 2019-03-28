''' Fer2013 Dataset class'''

from __future__ import print_function
from PIL import Image
import numpy as np
import h5py
import torch.utils.data as data
import cv2

class Mixed(data.Dataset):
    """`Mixed Dataset.

    Args:
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self, split='Training', filename='./data/data.h5', train_length=28709, validate_length=3589, test_length=3589, resize_length=96, image_channel=3,transform=None):
        self.transform = transform
        self.split = split  # training set or test set
        self.data = h5py.File(filename, 'r', driver='core')
        # now load the picked numpy arrays
        if self.split == 'Training':
            self.train_data = self.data['Training_pixel']
            self.train_labels = self.data['Training_label']
            self.train_data = np.asarray(self.train_data)
            self.train_data = self.train_data.reshape((train_length, resize_length, resize_length, image_channel))
            print('train finished!')

        elif self.split == 'PublicTest':
            self.PublicTest_data = self.data['PublicTest_pixel']
            self.PublicTest_labels = self.data['PublicTest_label']
            self.PublicTest_data = np.asarray(self.PublicTest_data)
            self.PublicTest_data = self.PublicTest_data.reshape((validate_length, resize_length, resize_length, image_channel))

        else:
            self.PrivateTest_data = self.data['PrivateTest_pixel']
            self.PrivateTest_labels = self.data['PrivateTest_label']
            self.PrivateTest_data = np.asarray(self.PrivateTest_data)
            self.PrivateTest_data = self.PrivateTest_data.reshape((test_length, resize_length, resize_length, image_channel))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split == 'Training':
            img, target = self.train_data[index], self.train_labels[index]
        elif self.split == 'PublicTest':
            img, target = self.PublicTest_data[index], self.PublicTest_labels[index]
        else:
            img, target = self.PrivateTest_data[index], self.PrivateTest_labels[index]

        # doing this so that it is consistent with all other datasets
        img_swap = img.swapaxes(0, 2)
        img = img_swap.swapaxes(1, 2)
        # img = Image.fromarray(img)
        img_merge = np.concatenate((img[0], img[1], img[2]))
        img_merge = Image.fromarray(img_merge)
        if self.transform is not None:
            img = self.transform(img_merge)
        return img, target

    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data)
        elif self.split == 'PublicTest':
            return len(self.PublicTest_data)
        else:
            return len(self.PrivateTest_data)

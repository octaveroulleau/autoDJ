"""This is the example module.

This module does stuff.
"""
# To manage Dataset (especially batch structure) on .npz data

import torch
from torch.utils.data import Dataset

import numpy as np


class NPZ_Dataset(Dataset):

    def __init__(self, npz_file, root_dir, dataName='Spectrums', transform=None):

        self.dataset_name = npz_file
        self.root_dir = root_dir
        self.path = self.root_dir + self.dataset_name

        npz_dict = np.load(self.path)

        # the data
        try:
            self.labels_stack = npz_dict['labels']
            self.imgs_stack = npz_dict[dataName]
        except:
            self.imgs_stack = npz_dict[dataName]

    # to support the indexing such that dataset[i] can be used to get ith
    # sample
    def __getitem__(self, idx):
        image = self.imgs_stack[:, idx]
        try:
            label = self.labels_stack[:, idx]
            singleData = {'image': image, 'label': label}
        except:
            singleData = {'image': image}
        return singleData

    # returns the size of the dataset
    def __len__(self):
        return len(self.imgs_stack[0, :])

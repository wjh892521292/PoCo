import torch
import torch.utils.data as data_utils
import numpy as np
import os
from PIL import Image, ImageDraw
import cv2
from xpinyin import Pinyin
from PIL import Image
import copy
import pandas as pd
import csv
import random


class MyDataset(data_utils.Dataset):

    def __init__(self, dataset, transform=None, fold=3):
        self.data_list = []
        self.items = []
        self.transform = transform

        if dataset == 'train':
            data_num = [i for i in range(5) if i != fold]
        elif dataset == 'valid':
            data_num = [fold]


        for i in data_num:
            f = open('/data2/chengyi/dataset/ord_reg/DR_dataset/ten_fold/fold_{}.csv'.format(i), "r")
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.items.append(row[1:])

      

    def __getitem__(self, idx):
        item = copy.deepcopy(self.items[idx])
        img = item[0]
        label = int(item[1])
        # img_path = '/data2/wangjinhong/data/ord_reg/DR_data/train/' + img + '.jpeg'
        img_path = '/data2/chengyi/dataset/ord_reg/DR_dataset/train/' + img + '.jpg'
        # label = int(item[-1])
        img = Image.open(img_path).convert('RGB')


        if label < 2:
            label = 0
        else:
            label = 1
            

        if self.transform:

            img = self.transform(img)
        
        return img, label  # l0, l1, l2#label

    def __len__(self):
        return len(self.items)

if __name__ == '__main__':
    for fold in range(10):
        d1 = MyDataset(None,None,'valid',fold=fold)
        d2 = MyDataset(None,None,'train',fold=fold)

        print(len(d1), len(d2))

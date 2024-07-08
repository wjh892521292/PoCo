
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


        f = open('/data/hhp/dataset/cataract_zs/Standardized_data/P_all_data.csv', "r")
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i%5 in data_num:
                self.items.append(row)
    
        print(self.items[0])
      

    def __getitem__(self, idx):
        item = copy.deepcopy(self.items[idx])
        img = item[0]
        label = int(item[1])
        img_path = '/data/hhp/dataset/cataract_zs/Standardized_data/image_data/' + img
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        print(img, label)
        return img, label  

    def __len__(self):
        return len(self.items)

if __name__ == '__main__':
    for fold in range(5):
        d = MyDataset(None,None,'valid',fold=fold)
        print(len(d))
    print(d[0])
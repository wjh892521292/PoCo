
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


        f = open('/data/hhp/dataset/cataract_zs/Standardized_data/C_all_data.csv', "r")
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i%5 in data_num:
                self.items.append(row)
    
      

    def __getitem__(self, idx):
        item = copy.deepcopy(self.items[idx])
        img = item[0]
        label = int(item[1])
        img_path = '/data/wangjinhong/data/cataract/ROI/crop_region_t/' + img
        img = Image.open(img_path).convert('RGB')
        
        img.show()

        if self.transform:
            img = self.transform(img)

        return img, label  

    def __len__(self):
        return len(self.items)

if __name__ == '__main__':
    for fold in range(5):
        d = MyDataset(None,None,'valid',fold=fold)
        print(len(d))

        for i in d:
            print(i)
            break
    print(d[0])
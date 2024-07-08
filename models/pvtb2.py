# import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from models.pvt import pvt_v2_b1, feature_pvt_v2_b2, feature_pvt_v2_b3

from models.basenet import BaseNet
from collections import OrderedDict
import models

class Model(nn.Module):

    def __init__(self, args):

        super(Model, self).__init__()
        self.dropout = args.dropout

        self.name = 'pvtb2'
        self.pvt = feature_pvt_v2_b3()
        ckpt = torch.load('/data2/chengyi/.torch/models/pvt_v2_b3.pth')
        self.pvt.load_state_dict(ckpt)

        self.out = nn.Linear(512, args.num_classes)

      

        self.dropout = args.dropout
        self.dp = torch.nn.Dropout(self.dropout)
        
    def model_name(self):
        return self.name

# make some changes to the end layer contrast to the original resnet
    def forward(self, x, tgt):

        x = self.pvt(x)

        
        x = x.permute(1, 0, 2)

        x = x.mean(dim=1)
        

        # x = self.relu(x)
        # if self.dropout < 1:
        #     x = self.dp(x)

        x = self.out(x)

        return x, nn.CrossEntropyLoss()(x, tgt.long())






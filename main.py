# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# from utils.data_load.data1 import MyDataset as data1
# from utils.data_load.data2 import MyDataset as data2

import torch
import torchvision.transforms as transforms
from config.cfg import BaseConfig
from training.trainer import DefaultTrainer
import os
from utils import data_load
import numpy as np
from module.Polar_transform import Polar_transform


###modality missing mask
masks = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]
masks_torch = torch.from_numpy(np.array(masks))
mask_name = ['t2', 't1c', 't1', 'flair', 
            't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
            'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
            'flairt1cet1t2']
print (masks_torch.int())

def main(args):
    runseed = 1
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    torch.manual_seed(runseed)
    np.random.seed(runseed)

    dataset = getattr(data_load, args.data_name.lower())
    
    train_data = dataset(
        dataset='train',
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            Polar_transform(args.polar),
#             transforms.RandomApply([
#                 transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
#             ], p=0.8),
#             transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        fold=args.k_fold
    )

    train_load = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                             drop_last=True)

    val_data = dataset(
        dataset='valid',
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            Polar_transform(args.polar),
#             transforms.RandomApply([
#                 transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
#             ], p=0.8),
#             transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        fold=args.k_fold
    )

    val_load = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

    trainer = DefaultTrainer(args)
    trainer.train(train_load, val_load)
    trainer.log_wrong()


if __name__ == '__main__':
    cfg = BaseConfig()
    fixed = None
    # if cfg.parser.parse_args().exp_name == 'test':
    #     fixed = '--exp_name test --model_name ctt_f --max_iter 1200  --stepvalues 600 900 1100  \
    #           --warmup_steps 100 --save_log /data2/wangjinhong/output/wjh/save_log/Cataract_OCT/logs_catar_test/ --batch_size 8 --display_freq 1 --val_freq 10 --lr 0.1 --z_dim 128 --resnet_layers 18  --dropout 0.5 --gpu_id 0'.split()
    args = cfg.initialize(fixed)
    main(args)
'--data_name aesthetics ' \
'--img_root /data2/wangjinhong/data/ord_reg/aesthetics/ ' \
'--data_root /data2/wangjinhong/data/ord_reg/beauty-icwsm15-dataset.tsv  ' \
 \
'''
cd /data2/chengyi/ord_reg
source activate torch18
python main.py

'''

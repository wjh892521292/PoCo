import os, sys
import cv2
import torch
import torch.nn as nn
import numpy as np
import models
from datetime import datetime
from tensorboardX import SummaryWriter
from sklearn.linear_model import LogisticRegression  

from config.cfg import arg2str
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
# from evaluater import metric



class DefaultTrainer(object):

    def __init__(self, args):
        self.args = args
        self.batch_size = args.batch_size
        self.lr = self.lr_current = args.lr
        self.start_iter = args.start_iter
        self.max_iter = args.max_iter
        self.warmup_steps = args.warmup_steps
        self.eval_only = args.eval_only
        self.model = getattr(models, args.model_name.lower())(args)
        self.model.cuda()
        self.loss = nn.CrossEntropyLoss()
        self.max_acc = 0
        self.tmp_idx_acc_with_mae = 0
        self.tmp_idx_acc_with_mae = 0
        self.min_loss = 1000
        self.max_auc = 0
        self.loss_name = args.loss_name
        self.start = 0
        self.wrong = None
        self.log_path = os.path.join(self.args.save_folder, self.args.exp_name, 'result.txt')
        self.optim = getattr(torch.optim, args.optim)(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, weight_decay=args.weight_decay)
        # self.log = open(self.log_path, mode='w')
        # self.log.write('============ ACC with MAE ============\n')
        # self.log.close()

        # if args.loss_name != 'POE':
        #     if self.args.optim == 'Adam':
        #         self.optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr,
        #                                       betas=(0.9, 0.999), eps=1e-08)
        #     else:
        #         self.optim = getattr(torch.optim, args.optim) \
        #             (filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, weight_decay=args.weight_decay)
        # else:
        #     # 这个只是用于vgg2的
        #     # print('LR = 0.0001')
        #     params = []
        #     for keys, param_value in self.model.named_parameters():
        #         if (is_fc(keys)):
        #             params += [{'params': [param_value], 'lr': 0.001}]
        #         else:
        #             params += [{'params': [param_value], 'lr': 0.0001}]
        #     #
        #     self.optim = torch.optim.Adam(params, lr=self.lr,
        #                                   betas=(0.9, 0.999), eps=1e-08)
        #
        # if args.resume:
        #     if os.path.isfile(self.args.resume):
        #         iter, index = self.load_model(args.resume)
        #         self.start_iter = iter

    def train_iter(self, step, dataloader):
        
        img, label = dataloader.next()
        img = img.float().cuda()
        label = label.cuda()
        self.model.eval()
        if self.eval_only:
            self.model.eval()

        pred, loss = self.model(img, label)
        # loss = self.loss(pred, label)

        '''generate logger'''
        if self.start == 0:
            self.init_writer()
            self.start = 1

        print('Training - Step: {} - Loss: {:.4f}' \
              .format(step, loss.item()))

        loss.backward()
        self.optim.step()
        self.model.zero_grad()

        if step % self.args.display_freq == 0:

            pred = pred.cpu().detach()
            pred2 = np.argmax(pred,axis=1)
            label = label.cpu().detach()

            acc = accuracy_score(label, pred2)
            recall = recall_score(label, pred2, average='macro')
            precision = precision_score(label, pred2, average='macro')
            f1 = f1_score(label, pred2, average='macro')


            # clf = LogisticRegression(solver="liblinear").fit(pred, label)

            # print(pred)
            # print(label)
            # print(clf.predict_proba(pred))
            # auc = roc_auc_score(label, clf.predict_proba(pred), average='macro', multi_class='ovr')
        

            print(
                'Training - Step: {} - Acc: {:.6f} - recall {:.6f} - precision {:.6f} - f1 {:.6f} - lr:{:.4f}' \
                    .format(step, acc, recall, precision, f1, self.lr_current))

            scalars = [loss.item(), acc, f1, self.lr_current]
            names = ['loss', 'acc', 'f1', 'lr']
            # scalars = [loss.item(), acc, mae, self.lr_current]
            # names = ['loss', 'acc', 'MAE', 'lr']
            write_scalars(self.writer, scalars, names, step, 'train')

    def train(self, train_dataloader, valid_dataloader=None):

        train_epoch_size = len(train_dataloader)
        train_iter = iter(train_dataloader)
        val_epoch_size = len(valid_dataloader)

        for step in range(self.start_iter, self.max_iter):

            if step % train_epoch_size == 0:
                print('Epoch: {} ----- step:{} - train_epoch size:{}'.format(step // train_epoch_size, step,
                                                                             train_epoch_size))
                train_iter = iter(train_dataloader)

            self._adjust_learning_rate_iter(step)
            self.train_iter(step, train_iter)

            if (valid_dataloader is not None) and (
                    step % self.args.val_freq == 0 or step == self.args.max_iter - 1) and (step != 0):
                val_iter = iter(valid_dataloader)
                val_loss, val_acc, val_auc = self.validation(step, val_iter, val_epoch_size)
                if val_acc > self.max_acc:
                    self.delete_model(best='best_acc', index=self.max_acc)
                    self.max_acc = val_acc
                    self.save_model(step, best='best_acc', index=self.max_acc, gpus=1)

                    # self.delete_model(best='best_acc_with_mae', index=self.tmp_idx_acc_with_mae)
                    # self.tmp_idx_acc_with_mae = [val_acc, val_mae]
                    # self.save_model(step, best='best_acc_with_mae', index=self.tmp_idx_acc_with_mae, gpus=1)
                    self.log = open(self.log_path, mode='a')
                    self.log.write('best_acc_with_auc = {}\n'.format([val_acc, val_auc]))
                    self.log.close()

                if val_loss.item() < self.min_loss:
                    self.delete_model(best='min_loss', index=self.min_loss)
                    self.min_loss = val_loss.item()
                    self.save_model(step, best='min_loss', index=self.min_loss, gpus=1)

                if val_auc.item() > self.max_auc:
                    self.delete_model(best='max_auc', index=self.max_auc)
                    self.max_auc = val_auc.item()
                    self.save_model(step, best='max_auc', index=self.max_auc, gpus=1)

                    # self.delete_model(best='min_mae_with_acc', index=self.tmp_idx_acc_with_mae)
                    # self.tmp_idx_mae_with_acc = [val_mae, val_acc]
                    # self.save_model(step, best='min_mae_with_acc', index=self.tmp_idx_acc_with_mae, gpus=1)
                    self.log = open(self.log_path, mode='a')
                    self.log.write('max_auc_with_acc = {}\n'.format([val_auc, val_acc]))
                    self.log.close()

        return self.min_loss, self.max_acc, self.max_auc
        # if step % self.args.save_freq == 0 and step != 0:
        #     self.model.save_model(step, best='step', index=step, gpus=1)

    def validation(self, step, val_iter, val_epoch_size):

        print('============Begin Validation============:step:{}'.format(step))

        self.model.eval()

        total_score = []
        total_target = []
        with torch.no_grad():
            for i in range(val_epoch_size):

                img, target = next(val_iter)
                img = img.float().cuda()
                target = target.cuda()

                score, loss = self.model(img, target)
                # score, loss = self.model(img, target)

                if i == 0:
                    total_score = score
                    total_target = target
                else:
                    if len(score.shape) == 1:
                        score = score.unsqueeze(0)
                   
                    total_score = torch.cat((total_score, score), 0)
                    total_target = torch.cat((total_target, target), 0)
        
        # print(total_score)
        # print(total_target)

        loss = self.loss(total_score, total_target)

        total_score = total_score.cpu().detach()
        total_score2 = np.argmax(total_score,axis=1)
        total_target = total_target.cpu().detach()

        acc = accuracy_score(total_score2, total_target)
        recall = recall_score(total_score2, total_target, average='macro')
        precision = precision_score(total_score2, total_target, average='macro')
        f1 = f1_score(total_score2, total_target, average='macro')

        if self.args.num_classes > 2:
            clf2 = LogisticRegression(solver="liblinear").fit(total_score, total_target)
            auc = roc_auc_score(total_target, clf2.predict_proba(total_score), average='macro', multi_class='ovr')
        else:
            clf2 = LogisticRegression(solver="liblinear", random_state=0).fit(total_score, total_target)
            auc = roc_auc_score(total_target, clf2.predict_proba(total_score)[:, 1])
                 

        '''
        记录做错的img
        '''
        # self.wrong_perspective_target = total_target.cpu().numpy()
        # _, pred = total_score.max(1)
        # wrong = (pred != total_target).float()
        # if self.wrong:
        #     self.wrong += wrong
        # else:
        #     self.wrong = wrong

        print(
            'Valid - Step: {} \n Loss: {:.4f} \n Acc: {:.6f} \n recall {:.6f} \n precision {:.6f} \n f1 {:.6f} \n auc {:.6f}' \
                    .format(step, loss, acc, recall, precision, f1, auc))
        scalars = [loss.item(), acc, precision, recall, f1, auc]
        names = ['loss', 'acc', 'precision', 'recall', 'f1score', 'auc']
        write_scalars(self.writer, scalars, names, step, 'val')

        return loss, acc, auc

    def log_wrong(self):
        # log = self.wrong
        pass
        # log = self.wrong.cpu().numpy()
        # # self.wrong_perspective_target
        # y = np.argsort(log)
        # tgts = self.wrong_perspective_target[y]
        # np.save("filename.npy", log)
        #
        # print('log:')
        # print(log[y][:20])
        # print('tgts:')
        # print(tgts[:20])
        # print('index:')
        # print(y[:20])

    ################

    def _adjust_learning_rate_iter(self, step):
        """Sets the learning rate to the initial LR decayed by 10 at every specified step
        # Adapted from PyTorch Imagenet example:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py
        """
        if step <= self.warmup_steps:  # 增大学习率
            self.lr_current = self.args.lr * float(step) / float(self.warmup_steps)

        if self.args.lr_adjust == 'fix':
            if step in self.args.stepvalues:
                self.lr_current = self.lr_current * self.args.gamma
        elif self.args.lr_adjust == 'poly':
            self.lr_current = self.args.lr * (1 - step / self.args.max_iter) ** 0.9

        for param_group in self.optim.param_groups:
            param_group['lr'] = self.lr_current

    def init_writer(self):
        """ Tensorboard writer initialization
            """

        if not os.path.exists(self.args.save_folder):
            os.makedirs(self.args.save_folder, exist_ok=True)

        if self.args.exp_name == 'test':
            log_path = os.path.join(self.args.save_log, self.args.exp_name)
        else:
            log_path = os.path.join(self.args.save_log,
                                    datetime.now().strftime('%b%d_%H-%M-%S') + '_' + self.args.optim + '_' + self.args.exp_name)
        log_config_path = os.path.join(log_path, 'configs.log')

        self.writer = SummaryWriter(log_path)
        with open(log_config_path, 'w') as f:
            f.write(arg2str(self.args))

    def load_model(self, model_path):
        if os.path.exists(model_path):
            load_dict = torch.load(model_path)
            net_state_dict = load_dict['net_state_dict']

            try:
                self.model.load_state_dict(net_state_dict)
            except:
                self.model.module.load_state_dict(net_state_dict)
            self.iter = load_dict['iter'] + 1
            index = load_dict['index']

            print('Model Loaded!')
            return self.iter, index
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

    def delete_model(self, best, index):
        if index == 0 or index == 1000000:
            return
        save_fname = '%s_%s_%s.pth' % (self.model.model_name(), best, index)
        save_path = os.path.join(self.args.save_folder, self.args.exp_name, save_fname)
        if os.path.exists(save_path):
            os.remove(save_path)

    def save_model(self, step, best='best_acc', index=None, gpus=1):

        model_save_path = os.path.join(self.args.save_folder, self.args.exp_name)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path, exist_ok=True)

        if gpus == 1:
            if isinstance(index, list):
                save_fname = '%s_%s_%s_%s.pth' % (self.model.model_name(), best, index[0], index[1])
            else:
                save_fname = '%s_%s_%s.pth' % (self.model.model_name(), best, index)
            save_path = os.path.join(self.args.save_folder, self.args.exp_name, save_fname)
            save_dict = {
                'net_state_dict': self.model.state_dict(),
                'exp_name': self.args.exp_name,
                'iter': step,
                'index': index
            }
        else:
            save_fname = '%s_%s_%s.pth' % (self.model.module.model_name(), best, index)
            save_path = os.path.join(self.args.save_folder, self.args.exp_name, save_fname)
            save_dict = {
                'net_state_dict': self.model.module.state_dict(),
                'exp_name': self.args.exp_name,
                'iter': step,
                'index': index
            }
        torch.save(save_dict, save_path)
        print(best + ' Model Saved')


def write_scalars(writer, scalars, names, n_iter, tag=None):
    for scalar, name in zip(scalars, names):
        if tag is not None:
            name = '/'.join([tag, name])
        writer.add_scalar(name, scalar, n_iter)

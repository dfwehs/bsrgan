import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np


import itertools
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from scipy.stats import entropy
from sklearn.metrics import confusion_matrix

import time

# The test method is the same as LISGAN.

def map_label(label, classes):
    mapped_label = torch.zeros(label.shape, dtype=torch.long).to(label.device)
    for i in range(classes.shape[0]):
        mapped_label[label==classes[i]] = i  
    return mapped_label

class CLASSIFIER:
    # Train_Y is interger
    def __init__(self, _train_X, _train_Y, data_loader, _nclass, _cuda, _lr=0.001, _beta1=0.5, _nepoch=20, _batch_size=100, generalized=True,ratio=0.6,epoch=20, cla=1,  device='cpu', logger=None):
        self.logger=logger
        self.train_X =  _train_X.cpu() 
        self.train_Y = _train_Y.cpu() 
        ##########################################################
        self.test_seen_feature = data_loader.test_seen_feature
        self.test_seen_label = data_loader.test_seen_label 
        self.test_unseen_feature = data_loader.test_unseen_feature
        self.test_unseen_label = data_loader.test_unseen_label 
        self.seenclasses = data_loader.seenclasses
        self.unseenclasses = data_loader.unseenclasses
        self.device=device
        ##########################################################
        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.nclass = _nclass
        self.input_dim = _train_X.size(1)
        self.cuda = _cuda
        if cla==1:
            self.model = LINEAR_LOGSOFTMAX(self.input_dim, self.nclass)
        else:
            self.model = LINEAR_LOGSOFTMAX1(self.input_dim, self.nclass)
        self.model.apply(weights_init)
        self.criterion = nn.NLLLoss()
        
        
        self.lr = _lr
        self.beta1 = _beta1
        # setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=_lr, betas=(_beta1, 0.999))
        self.ratio = ratio
        self.epoch = epoch

        self.model.to(self.device)
        self.criterion.to(self.device)


        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.size()[0]
        self.backup_X = _train_X.cpu()
        self.backup_Y = _train_Y.cpu()

        if generalized:
            self.acc_seen, self.acc_unseen, self.H = self.fit()
        else:
            self.acc = self.fit_zsl()

    def pairwise_distances(self,x, y=None):
        '''
        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        '''
        x_norm = (x ** 2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y ** 2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)
        
        
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        # Ensure diagonal is zero if x=y
        if y is None:
            dist = dist - torch.diag(dist.diag)
        return torch.clamp(dist, 0.0, np.inf)

    def fit_zsl(self):
        first_acc=0
        first_all_pred = None
        first_all_output = None
        first_all_acc = []

        trun = lambda x: int(x * 100)

        all_length = self.test_unseen_feature.size(0)
        mapped_test_label = map_label(self.test_unseen_label, self.unseenclasses)
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)
                inputv = batch_input.detach().to(self.device)
                labelv = batch_label.detach().to(self.device)
                output = self.model(inputv)
                loss = self.criterion(output, labelv)  # 使用fake_unseen_feature和labels来训练分类器
                loss.backward()
                self.optimizer.step()

            acc, pred, output,all_acc = self.val(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
            if acc > first_acc:
                first_acc = acc
                first_all_pred = pred
                first_all_output = output
                first_all_acc = all_acc

        self.logger.log('SF a: %.2f%%' % (first_acc * 100))
		
        easy_len = int(all_length*self.ratio)
        hard_len = all_length - easy_len
        entropy_value = torch.from_numpy(np.asarray(list(map(entropy,first_all_output.data.cpu()))))
        _, indices = torch.sort(-entropy_value)
        exit_indices = indices[:easy_len]
        keep_indices = indices[easy_len:]

        first_easy_pred = first_all_pred[exit_indices]
        first_easy_label = mapped_test_label[exit_indices]
        first_hard_pred = first_all_pred[keep_indices]
        first_hard_label = mapped_test_label[keep_indices]
        acc_first_easy = self.compute_per_class_acc(first_easy_label,first_easy_pred,self.unseenclasses.size(0))
        acc_first_hard = self.compute_per_class_acc(first_hard_label,first_hard_pred,self.unseenclasses.size(0))
        all_easy_hard_label = torch.cat( (first_easy_label,first_hard_label),0 )

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.backup_X.size()[0] + easy_len # fake+easy_sample

        self.train_X = torch.cat( (self.backup_X, self.test_unseen_feature[exit_indices.cpu()] ),0 )
        self.train_Y = torch.cat( (self.backup_Y, first_easy_pred.cpu() ),0 )

        sims = self.pairwise_distances(self.test_unseen_feature[keep_indices.cpu()], self.train_X)
        value,idx = torch.min(sims,dim=1)
        knn_hard_pred = self.train_Y[idx]
        
        knn_all_pred = torch.cat( (first_easy_pred, knn_hard_pred),0 )

        acc_knn_hard = self.compute_per_class_acc(first_hard_label, knn_hard_pred,self.unseenclasses.size(0))
        acc_knn = self.compute_per_class_acc(all_easy_hard_label,knn_all_pred,self.unseenclasses.size(0))
        all_acc_knn = self.compute_every_class_acc(all_easy_hard_label,knn_all_pred,self.unseenclasses.size(0))

        self.logger.log('NN a: %.2f%%' % (acc_knn*100))

        acc_fc_hard = 0
        fc_hard_pred = None
        real_mean_acc = 0
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)

                inputv = batch_input.detach().to(self.device)
                labelv = batch_label.detach().to(self.device)
                output = self.model(inputv)
                loss = self.criterion(output, labelv)
                loss.backward()
                self.optimizer.step()
            acc, pred, output,_ = self.val(self.test_unseen_feature[keep_indices],self.test_unseen_label[keep_indices],self.unseenclasses)


            correct_num = pred.eq(mapped_test_label[keep_indices]).cpu().sum().float()
            all_length = output.size(0)
            first_acc = correct_num / all_length
            

            if first_acc > acc_fc_hard:
                acc_fc_hard = acc
                fc_hard_pred = pred
                real_mean_acc = first_acc
        fc_all_pred = torch.cat( [first_easy_pred,fc_hard_pred],0 )
        acc_fc_hard = self.compute_per_class_acc(first_hard_label, fc_hard_pred,self.unseenclasses.size(0))
        acc_fc = self.compute_per_class_acc(all_easy_hard_label,fc_all_pred,self.unseenclasses.size(0))
        all_acc_fc = self.compute_every_class_acc(all_easy_hard_label,fc_all_pred,self.unseenclasses.size(0))
        
        self.logger.log('FC a: %.2f%%' % (acc_fc*100))

        sys.stdout.flush()
        return acc_fc



    def split_pred(self,all_pred, real_label):
        seen_pred = None
        seen_label = None
        unseen_pred = None
        unseen_label = None
        for i in self.seenclasses:
            idx = (real_label == i)
            if seen_pred is None:
                seen_pred = all_pred[idx]
                seen_label = real_label[idx]
            else:
                seen_pred = torch.cat( (seen_pred,all_pred[idx]),0 )
                seen_label = torch.cat( (seen_label, real_label[idx]) )

        for i in self.unseenclasses:
            idx = (real_label == i)
            if unseen_pred is None:
                unseen_pred = all_pred[idx]
                unseen_label = real_label[idx]
            else:
                unseen_pred = torch.cat( (unseen_pred,all_pred[idx]),0 )
                unseen_label = torch.cat(  (unseen_label, real_label[idx]), 0 )

        return seen_pred, seen_label, unseen_pred, unseen_label



    # for gzsl
    def fit(self):
        # 3个length
        test_seen_length = self.test_seen_feature.shape[0] #1764
        test_unseen_length = self.test_unseen_feature.shape[0] #2967
        all_length = test_seen_length + test_unseen_length
        all_test_feature = torch.cat( (self.test_seen_feature,self.test_unseen_feature), 0 )
        all_test_label = torch.cat( (self.test_seen_label, self.test_unseen_label), 0 )
        all_classes = torch.sort(torch.cat( (self.seenclasses,self.unseenclasses),0 ))[0]
        first_acc = 0
        first_all_pred = None
        first_all_output = None

        best_H = 0
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size): #self.ntrain=22057, self.batch_size=300
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)

                inputv = batch_input.detach().to(self.device)
                labelv = batch_label.detach().to(self.device)
                output = self.model(inputv)
                loss = self.criterion(output, labelv)
                loss.backward()
                self.optimizer.step()

            acc_seen,pred_seen,output_seen = self.val_gzsl(self.test_seen_feature, self.test_seen_label, self.seenclasses)
            acc_unseen,pred_unseen,output_unseen = self.val_gzsl(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
            H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
            if H >= best_H:
                best_H = H
                first_all_pred = torch.cat( (pred_seen,pred_unseen), 0 )
                first_all_output = torch.cat( (output_seen, output_unseen), 0 )

        first_seen_pred,first_seen_label,first_unseen_pred,first_unseen_label = self.split_pred(first_all_pred,all_test_label)

        acc_first_seen = self.compute_per_class_acc_gzsl(first_seen_label, first_seen_pred,self.seenclasses)
        acc_first_unseen = self.compute_per_class_acc_gzsl(first_unseen_label, first_unseen_pred,self.unseenclasses)
        acc_first_H = 2*acc_first_seen*acc_first_unseen/(acc_first_seen+acc_first_unseen)
        self.logger.log('SF s: %.2f%%, u: %.2f%%, h: %.2f%%' % (acc_first_seen*100,acc_first_unseen*100,acc_first_H*100))
  

        easy_length = int(all_length*self.ratio)
        hard_length = all_length - easy_length
        entropy_value = torch.from_numpy(np.asarray(list(map(entropy, first_all_output.data.cpu()))))
        _, indices = torch.sort(-entropy_value)
        exit_indices = indices[:easy_length]
        keep_indices = indices[easy_length:]
        first_easy_pred = first_all_pred[exit_indices]
        first_easy_label = all_test_label[exit_indices]
        first_hard_pred = first_all_pred[keep_indices]
        first_hard_label = all_test_label[keep_indices]
        all_easy_hard_label = torch.cat( (first_easy_label,first_hard_label),0 )

        acc_first_easy = self.compute_per_class_acc_gzsl(first_easy_label,first_easy_pred,all_classes)
        acc_first_hard = self.compute_per_class_acc_gzsl(first_hard_label,first_hard_pred,all_classes)

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.backup_X.size(0) + easy_length
        first_easy_pred = first_easy_pred.cpu()
        

        self.train_X = torch.cat( (self.backup_X, all_test_feature[exit_indices]),0 )
        self.train_Y = torch.cat( (self.backup_Y, first_easy_pred),0)
        
        sims = self.pairwise_distances(all_test_feature[keep_indices], self.train_X)
        value, idx = torch.min(sims, dim=1)
        knn_hard_pred = self.train_Y[idx]
        knn_all_pred = torch.cat( (first_easy_pred,knn_hard_pred),0 )
        knn_seen_pred,knn_seen_label,knn_unseen_pred,knn_unseen_label = self.split_pred(knn_all_pred,all_easy_hard_label)
        acc_knn_seen = self.compute_per_class_acc_gzsl(knn_seen_label,knn_seen_pred,self.seenclasses)
        acc_knn_unseen = self.compute_per_class_acc_gzsl(knn_unseen_label,knn_unseen_pred,self.unseenclasses)
        acc_knn_H = 2*acc_knn_seen*acc_knn_unseen/(acc_knn_seen+acc_knn_unseen)
        self.logger.log('NN s: %.2f%%, u: %.2f%%, h: %.2f%%' % (acc_knn_seen*100,acc_knn_unseen*100,acc_knn_H*100))

        best_fc_hard_acc = 0
        fc_hard_pred = None
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)
                inputv = batch_input.detach().to(self.device)
                labelv = batch_label.detach().to(self.device)
                output = self.model(inputv)
                loss = self.criterion(output, labelv.to(self.device))  # 使用fake_unseen_feature和labels来训练分类器
                loss.backward()
                self.optimizer.step()

            acc,pred,_ = self.val_gzsl(all_test_feature[keep_indices],first_hard_label,all_classes)
            if acc > best_fc_hard_acc:
                fc_hard_pred = pred

        fc_all_pred = torch.cat((first_easy_pred, fc_hard_pred), 0)
        fc_seen_pred, fc_seen_label, fc_unseen_pred, fc_unseen_label = self.split_pred(fc_all_pred,all_easy_hard_label)
        acc_fc_seen = self.compute_per_class_acc_gzsl(fc_seen_label, fc_seen_pred, self.seenclasses)
        acc_fc_unseen = self.compute_per_class_acc_gzsl(fc_unseen_label, fc_unseen_pred, self.unseenclasses)
        acc_fc_H = 2 * acc_fc_seen * acc_fc_unseen / (acc_fc_seen + acc_fc_unseen)
        self.logger.log('FC s: %.2f%%, u: %.2f%%, h: %.2f%%' % (acc_fc_seen * 100,acc_fc_unseen * 100,acc_fc_H * 100))

        acc_fc_hard = self.compute_per_class_acc_gzsl(first_hard_label,fc_hard_pred,all_classes)

        sys.stdout.flush()
        return acc_fc_seen,acc_fc_unseen,acc_fc_H

    def val(self, test_X, test_label, target_classes,second=False):
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        all_output = None
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)

            with torch.no_grad():
                output = self.model(test_X[start:end].to(self.device))

            if all_output is None:
                all_output = output
            else:
                all_output = torch.cat( (all_output, output), 0 )
            _, predicted_label[start:end] = torch.max(output.data, 1)
            start = end
        acc = self.compute_per_class_acc(map_label(test_label, target_classes), predicted_label, target_classes.size(0))
        acc_all = self.compute_every_class_acc(map_label(test_label, target_classes), predicted_label, target_classes.size(0))
        return acc, predicted_label, all_output,acc_all

    def val_gzsl(self, test_X, test_label, target_classes):
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        all_output = None
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)

            with torch.no_grad():
                output = self.model(test_X[start:end].to(self.device))


            if all_output is None:
                all_output = output
            else:
                all_output = torch.cat( (all_output, output), 0 )
            _, predicted_label[start:end] = torch.max(output.data, 1)
            start = end
        acc = self.compute_per_class_acc_gzsl(test_label, predicted_label, target_classes)
        return acc, predicted_label, all_output

    def next_batch(self, batch_size):
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
            # shuffle the data
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]
            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0) , torch.cat((Y_rest_part, Y_new_part), 0)
            else:
                return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            return self.train_X[start:end], self.train_Y[start:end]

    def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes):
        acc_per_class = 0
        for i in target_classes:
            idx = (test_label == i)
            if torch.sum(idx) == 0:
                continue
            else:
                acc_per_class += torch.sum(test_label[idx]==predicted_label[idx]).float() / torch.sum(idx).float()
        acc_per_class /= target_classes.size(0)
        return acc_per_class 

    def compute_per_class_acc(self, test_label, predicted_label, nclass):
        acc_per_class = torch.FloatTensor(nclass).fill_(0)
        for i in range(nclass):
            idx = (test_label == i)
            if torch.sum(idx) != 0:
                acc_per_class[i] = torch.sum(test_label[idx]==predicted_label[idx]).float() / torch.sum(idx).float()
        return acc_per_class.mean()

    def compute_every_class_acc(self, test_label, predicted_label, nclass):
        acc_per_class = torch.FloatTensor(nclass).fill_(0)
        for i in range(nclass):
            idx = (test_label == i)
            if torch.sum(idx) != 0:
                acc_per_class[i] = torch.sum(test_label[idx]==predicted_label[idx]).float() / torch.sum(idx).float()
        return acc_per_class

class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.model = nn.Sequential(nn.Linear(input_dim, nclass),
                                  nn.LogSoftmax(dim=1))
    def forward(self, x): 
        o = self.model(x)
        return o  

    
class LINEAR_LOGSOFTMAX1(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX1, self).__init__()
        self.model = nn.Sequential(nn.Linear(input_dim, 4096),
                                  nn.ReLU(True),
                                  nn.Linear(4096, nclass),
                                  nn.LogSoftmax(dim=1))
    def forward(self, x): 
        o = self.model(x)
        return o  

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


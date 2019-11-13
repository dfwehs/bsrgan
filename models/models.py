import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import torchvision
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import imshow
import scipy.io as sio
import sklearn.preprocessing as preprocessing
import yaml
import argparse

    
    
class CLA(nn.Module):
    def __init__(self, f_dim,  class_num, h_dim=512):
        super(CLA,self).__init__()
        self.model = nn.Sequential(nn.Linear(f_dim, h_dim),
                                   nn.ReLU(),
                                   nn.Linear(h_dim, class_num),
                                   nn.LogSoftmax(dim=1)
                                   )
        self.apply(weights_init)
    def forward(self,x):
        return self.model(x)
    

# 将label映射到可以计算交叉熵的结果
def map_label(label, classes):
    mapped_label = torch.zeros(label.shape, dtype=torch.long).to(label.device)
    for i in range(classes.shape[0]):
        mapped_label[label==classes[i]] = i  
    return mapped_label


class train_cla():
    def __init__(self, train_data, train_target, batch_size=64, device='cpu'):
        self.device = device
        self.classes = torch.unique(train_target).to(self.device)
        self.train_target = train_target.to(self.device)
        self.train_data = train_target.to(self.device)
        self.data = MyDataset(train_data.type(torch.float),map_label(self.train_target, self.classes).type(torch.long),shuffle=True, batch_size=batch_size).to(self.device)
        self.cla = CLA(train_data.shape[1], torch.unique(train_target).shape[0]).to(self.device)
        self.criterion = nn.NLLLoss().to(self.device)
        self.optimizer = optim.Adam(self.cla.parameters(), lr=1e-3,weight_decay=1e-8)
    
    def test_cla(self, cla, test_data, test_target, batch_size=64):
        test_data = test_data.to(self.device)
        test_target = test_target.to(self.device)
        test_data = MyDataset(test_data,map_label(test_target, self.classes),shuffle=False,batch_size=batch_size).to(self.device)
        acc = 0
        acount = 0
        while not test_data.isEnded():
            x, y = test_data.next_batch()
            pred = cla(x)
            res=torch.max(pred, dim=1)[1]
            res = (res==y).type(torch.float).sum()/x.shape[0]
            acc = acc + res
            acount = acount + 1
        acc = acc / acount
        return acc.item()
    
    def train(self):
        self.data.reset()
        while(not self.data.isEnded()):
            x, y = self.data.next_batch()
            pred = self.cla(x)
            loss = self.criterion(pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def run(self, epoch, test_data, test_target, batch_size=64, save_path="./cla_model"):
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        best_acc = 0
        for i in range(epoch):
            self.train()
            res=self.test_cla(self.cla, test_data, test_target, batch_size)
            if res>best_acc:
                best_acc = res
                torch.save(self.cla, save_path+"/model.pt")
        print("Besc cla acc:%.4f" % best_acc)


class Discriminator(nn.Module):
    def __init__(self, opts):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(nn.Linear(opts.f_dim + opts.atts_dim, opts.h_dim),
                                   nn.ReLU(True),
                                   nn.Linear(opts.h_dim, 1))
        self.apply(weights_init)
    
    def forward(self, x, att):
        h = torch.cat([x, att], 1)
        return self.model(h)

class Discriminator1(nn.Module):
    def __init__(self, opts):
        super(Discriminator1, self).__init__()
        self.model = nn.Sequential(nn.Linear(opts.f_dim + opts.atts_dim, opts.h_dim),
                                   nn.LeakyReLU(0.2, True),
                                   nn.Linear(opts.h_dim, 1),
                                   nn.ReLU(True))
        self.apply(weights_init)
    
    def forward(self, x, att):
        h = torch.cat([x, att], 1)
        return self.model(h)
    
class Generator(nn.Module):
    def __init__(self, opts):
        super(Generator, self).__init__()
        self.model = nn.Sequential(nn.Linear(opts.atts_dim + opts.nz, opts.h_dim),
                                   nn.ReLU(True),
                                   nn.Linear(opts.h_dim, opts.f_dim))
        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat([noise, att], 1)
        return self.model(h)
    
class Generator1(nn.Module):
    def __init__(self, opts):
        super(Generator1, self).__init__()
        self.model = nn.Sequential(nn.Linear(opts.atts_dim + opts.nz, opts.h_dim),
                                   nn.LeakyReLU(0.2, True),
                                   nn.Linear(opts.h_dim, opts.f_dim),
                                   nn.ReLU(True))
        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat([noise, att], 1)
        return self.model(h)
    
class Reconstructor(nn.Module):
    def __init__(self, opts):
        super(Reconstructor, self).__init__()
        self.model = nn.Sequential(nn.Linear(opts.f_dim, opts.h_dim),
                                   nn.ReLU(True),
                                   nn.Linear(opts.h_dim, opts.atts_dim),
                                   )
        self.apply(weights_init)

    def forward(self, f):
        return self.model(f)   
		
		
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def calc_gradient_penalty(netD, real_data, fake_data, input_att, opts):
    # print real_data.size()
    alpha = torch.rand(opts.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(opts.device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.to(opts.device)

    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates, Variable(input_att))

    ones = torch.ones(disc_interpolates.size())
    ones = ones.to(opts.device)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opts.lambda1
    return gradient_penalty

# packager
class MyDataset():
    def  __init__(self, data, target, shuffle=True, batch_size=64):
        self.data = data
        self.target = target
        self.ntrain = self.data.shape[0]
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.max_length = data.shape[0]
        self.start_index = 0
        
        self.ended = False
        
        if(self.shuffle==True):
            self._shuffle()
        
    
    def _shuffle(self):
        p = np.arange(self.max_length)
        np.random.shuffle(p)
        self.data = self.data[p]
        self.target = self.target[p]
        
    def next_batch(self):
        if(self.shuffle==False and self.ended==True):
            raise Exception("Data iter ended!")
        end_index = self.start_index + self.batch_size
        if end_index > self.max_length:
            if self.start_index < self.max_length:
                end_index = self.max_length
                self.ended = True
            else:
                self.start_index=0
                end_index = self.start_index + self.batch_size
                if(self.shuffle == True):
                    self._shuffle()
        if end_index > self.max_length:
            raise Exception("No more data")
        data = self.data[self.start_index:end_index]
        target = self.target[self.start_index:end_index]

        self.start_index = end_index
        return data, target
    
    def reset(self):
        self.start_index = 0
        self.ended = False
        if(self.shuffle == True):
            self._shuffle()
            
    def isEnded(self):
        return self.ended
    
    def to(self, device):
        self.data = self.data.to(device)
        self.target = self.target.to(device)
        return self


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

class Logger:
    def __init__(self, path):
        self.path=path
    
    def log(self, res):
        print(str(res))
        with open(self.path, mode='a') as f:
            f.write(str(res))
            f.write('\n') # 换行

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed) #cpu
    torch.cuda.manual_seed_all(seed)  #并行gpu
    torch.backends.cudnn.deterministic = True  #cpu/gpu结果一致
    torch.backends.cudnn.benchmark = True   #训练集变化不大时使训练加速

def load_yaml(file_name):
    with open(file_name, 'r') as f:
        file_data = f.read()
    res = yaml.load(file_data)
    opts = argparse.Namespace()
    opts.f_dim = res['f_dim']
    opts.atts_dim = res['atts_dim']
    opts.h_dim = res['h_dim']
    opts.nz = res['nz']
    opts.device = res['device']
    opts.batch_size = res['batch_size']
    opts.lambda1 = res['lambda1']
    opts.lr = res['lr']
    opts.r_lr = res['r_lr']
    opts.beta1 = res['beta1']
    opts.nepoch = res['nepoch']
    opts.critic_iter = res['critic_iter']
    opts.cls_weight = res['cls_weight']
    opts.syn_num = res['syn_num']
    opts.syn_num_s = res['syn_num_s']
    opts.classifier_lr = res['classifier_lr']
    opts.ratio = res['ratio']
    opts.random_seed = res['random_seed']
    opts.dataroot = res['dataroot']
    opts.dataset = res['dataset']
    opts.image_embedding = res['image_embedding']
    opts.class_embedding = res['class_embedding']
    opts.nclass_all = res['nclass_all']
    opts.cla_epoch = res['cla_epoch']
    opts.gzsl = res['gzsl']
    opts.zsl = res['zsl']
    opts.gzsl_model_path = res['gzsl_model_path']
    opts.zsl_model_path = res['zsl_model_path']
    opts.preprocessing = res['preprocessing']
    opts.sematic_cla =res['sematic_cla']
    opts.cla = res['cla']
    opts.GD = res['GD']
    opts.cla_syn_num = res['cla_syn_num']
    opts.optimizer=res['optimizer']
    opts.lambda2=res['lambda2']
    return opts


# 将数据打包
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
    
    

# 将label映射到可以计算交叉熵的结果
def map_label(label, classes):
    mapped_label = torch.zeros(label.shape, dtype=torch.long).to(label.device)
    for i in range(classes.shape[0]):
        mapped_label[label==classes[i]] = i  
    return mapped_label


class train_cla():
    def __init__(self, train_data, train_target, CLA, batch_size=64, device='cpu'):
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
#         print("Besc cla acc:%.4f" % best_acc)


    
class DATA_LOADER(object):
    def __init__(self, opt):

        self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0

        self.feature_dim = self.train_feature.shape[1]
        self.att_dim = self.attribute.shape[1]
        self.text_dim = self.att_dim
        self.train_cls_num = self.seenclasses.shape[0]
        self.test_cls_num = self.unseenclasses.shape[0]
        self.tr_cls_centroid = np.zeros([self.seenclasses.shape[0], self.feature_dim], np.float32)
        for i in range(self.seenclasses.shape[0]):
            self.tr_cls_centroid[i] = np.mean(self.train_feature[torch.nonzero(self.train_mapped_label == i),:].numpy(), axis=0)





    def read_matdataset(self, opt):
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
#         train_loc = matcontent['train_loc'].squeeze() - 1
#         val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

        self.attribute = torch.from_numpy(matcontent['att'].T).float()

        #############数据标准化###############
        if (opt.preprocessing):
            scaler = preprocessing.MinMaxScaler()
            _train_feature = scaler.fit_transform(feature[trainval_loc])  # feature[trainval_loc]
            _test_seen_feature =  scaler.transform(feature[test_seen_loc]) # feature[test_seen_loc]
            _test_unseen_feature = scaler.transform(feature[test_unseen_loc]) # feature[test_unseen_loc] 
            self.train_feature = torch.from_numpy(_train_feature).float()
            mx = self.train_feature.max() # 1
        else:
            _train_feature = feature[trainval_loc]
            _test_seen_feature = feature[test_seen_loc]
            _test_unseen_feature = feature[test_unseen_loc] 
            self.train_feature = torch.from_numpy(_train_feature).float()
            mx = 1
        self.train_feature.mul_(1/mx)
        self.train_label = torch.from_numpy(label[trainval_loc]).long()
        self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
        self.test_unseen_feature.mul_(1/mx)
        self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
        self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
        self.test_seen_feature.mul_(1/mx)
        self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
        #####################################
        
        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.ntrain = self.train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()
        self.train_mapped_label = map_label(self.train_label, self.seenclasses)
        self.train_att = self.attribute[self.seenclasses].numpy()
        self.test_att  = self.attribute[self.unseenclasses].numpy()
        self.train_cls_num = self.ntrain_class
        self.test_cls_num  = self.ntest_class

    def next_batch_one_class(self, batch_size):
        if self.index_in_epoch == self.ntrain_class:
            self.index_in_epoch = 0 
            perm = torch.randperm(self.ntrain_class)
            self.train_class[perm] = self.train_class[perm]

        iclass = self.train_class[self.index_in_epoch]
        idx = self.train_label.eq(iclass).nonzero().squeeze()
        perm = torch.randperm(idx.size(0))
        idx = idx[perm]
        iclass_feature = self.train_feature[idx]
        iclass_label = self.train_label[idx]
        self.index_in_epoch += 1
        return iclass_feature[0:batch_size], iclass_label[0:batch_size], self.attribute[iclass_label[0:batch_size]] 
    
    def next_batch(self, batch_size):
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_feature, batch_label, batch_att

    # select batch samples by randomly drawing batch_size classes    
    def next_batch_uniform_class(self, batch_size):
        batch_class = torch.LongTensor(batch_size)
        for i in range(batch_size):
            idx = torch.randperm(self.ntrain_class)[0]
            batch_class[i] = self.train_class[idx]
            
        batch_feature = torch.FloatTensor(batch_size, self.train_feature.size(1))       
        batch_label = torch.LongTensor(batch_size)
        batch_att = torch.FloatTensor(batch_size, self.attribute.size(1))
        for i in range(batch_size):
            iclass = batch_class[i]
            idx_iclass = self.train_label.eq(iclass).nonzero().squeeze()
            idx_in_iclass = torch.randperm(idx_iclass.size(0))[0]
            idx_file = idx_iclass[idx_in_iclass]
            batch_feature[i] = self.train_feature[idx_file]
            batch_label[i] = self.train_label[idx_file]
            batch_att[i] = self.attribute[batch_label[i]] 
        return batch_feature, batch_label, batch_att

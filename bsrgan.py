import os
import numpy as np
import scipy
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
from models.models import Reconstructor, Generator, Discriminator, Generator1, Discriminator1, CLA, calc_gradient_penalty
import models.classifier as classifier
import argparse
import sklearn.preprocessing as preprocessing
from scipy.stats import entropy
from sklearn.metrics import confusion_matrix
import time
import itertools
import sys
import utils
from termcolor import cprint
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--c', default='./config/sun_gzsl.yaml', help='config yaml')
config_path = parser.parse_args().c

opts=utils.load_yaml(config_path)
utils.setup_seed(opts.random_seed)
data = utils.DATA_LOADER(opts)
logger=None

if opts.zsl:
    model_save_path = './results/' + opts.zsl_model_path
else:
    model_save_path = './results/' + opts.gzsl_model_path

if not os.path.exists('./results'):
    os.mkdir('results')

if opts.zsl:
    logger = utils.Logger('./results/' + opts.zsl_model_path + '/result.log')
    if not os.path.exists('./results/' + opts.zsl_model_path):
        os.mkdir('./results/' + opts.zsl_model_path)


if opts.gzsl:
    logger = utils.Logger('./results/' + opts.gzsl_model_path + '/result.log')
    if  not os.path.exists('./results/' + opts.gzsl_model_path):
        os.mkdir('./results/' + opts.gzsl_model_path)


logger.log("starting in %s..." % (opts.dataset))


def generate_syn_feature_att(netG, classes, attribute, num, device="cpu"):
    nclass = classes.size(0)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(nclass * num, opts.atts_dim)
    syn_noise = torch.FloatTensor(nclass * num, opts.nz)
    
    syn_att = syn_att.to(device)
    syn_noise = syn_noise.to(device)
    
    syn_noise.normal_(0, 1)
    
    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.narrow(0, i * num, num).copy_(iclass_att.repeat(num, 1))
        syn_label.narrow(0, i * num, num).fill_(iclass)
    
    with torch.no_grad():
        syn_feature = netG(syn_noise, syn_att)
    
    a = syn_feature.detach().cpu()
    b = syn_label.detach().cpu()
    c = syn_att.detach().cpu()
    del syn_feature,syn_label,syn_att
    return a,b,c



def generate_syn_feature_with_grad_and_att(netG, classes, attribute, num):
    nclass = classes.size(0)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(nclass * num, opts.atts_dim)
    syn_noise = torch.FloatTensor(nclass * num, opts.nz)
    
    syn_att = syn_att.to(opts.device)
    syn_noise = syn_noise.to(opts.device)
    
    syn_noise.normal_(0, 1)
    
    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.narrow(0, i * num, num).copy_(iclass_att.repeat(num, 1))
        syn_label.narrow(0, i * num, num).fill_(iclass)
    
    syn_feature = netG(syn_noise, syn_att)

    return syn_feature, syn_label, syn_att

""""pre-train a classifier on seen classes"""
trc = utils.train_cla(data.train_feature, data.train_label, CLA, device=opts.device, )
trc.run(50, data.test_seen_feature, data.test_seen_label, save_path='./cla_model')
# load best classifier
pre_cla = torch.load("./cla_model/model.pt")


for p in pre_cla.parameters():  # set requires_grad to False
    p.requires_grad = False
    
if(opts.GD == 1):
    netG = Generator(opts).to(opts.device)
    netD = Discriminator(opts).to(opts.device)
else:
    netG = Generator1(opts).to(opts.device)
    netD = Discriminator1(opts).to(opts.device)

# seen reconstructor
netRS = Reconstructor(opts).to(opts.device)
# unseen reconstructor
netRU = Reconstructor(opts).to(opts.device)

if opts.optimizer == "ADAM":
    optimzerF = optim.Adam
else:
    optimzerF = optim.RMSprop

#train setup
optimizerD = optimzerF(netD.parameters(), lr=opts.lr)
optimizerG = optimzerF(netG.parameters(), lr=opts.lr)
optimizerRS = optimzerF(netRS.parameters(), lr=opts.r_lr)
optimizerRU = optimzerF(netRU.parameters(), lr=opts.r_lr)

cls_criterion = nn.NLLLoss().to(opts.device)
mse_criterion = nn.MSELoss().to(opts.device)
noise = torch.FloatTensor(opts.batch_size, opts.nz).to(opts.device)
input_res = torch.FloatTensor(opts.batch_size, opts.f_dim).to(opts.device)
input_att = torch.FloatTensor(opts.batch_size, opts.atts_dim).to(opts.device)
input_label = torch.LongTensor(opts.batch_size).to(opts.device)


# training and test
seenclasses = data.seenclasses.to(opts.device)
unseenclasses = data.unseenclasses.to(opts.device)
for epoch in range(opts.nepoch):
    netRS.to(opts.device)
    netRU.to(opts.device)
    for i in range(0, data.ntrain, opts.batch_size):
        netD.train()
        netG.eval()
        # train Discriminator
        for iter_d in range(opts.critic_iter):
            batch_feat, batch_l, batch_att = data.next_batch(opts.batch_size)
            batch_feat = batch_feat.to(opts.device)
            batch_l = batch_l.to(opts.device)
            batch_att = batch_att.to(opts.device)

            netD.zero_grad()
            
            # real loss
            criticD_real = netD(batch_feat, batch_att)
            criticD_real = - criticD_real.mean()
            criticD_real.backward()

            # fake loss
            noise.normal_(0, 1)
            fake = netG(noise, batch_att)
            fake_norm = fake.data[0].norm()

            criticD_fake = netD(fake.detach(), batch_att)
            criticD_fake = criticD_fake.mean()
            criticD_fake.backward()
            
            # gradient penalty loss
            gradient_penalty = calc_gradient_penalty(netD, batch_feat, fake.data, batch_att, opts)
            gradient_penalty.backward()

            Wasserstein_D = criticD_real - criticD_fake
            D_cost = criticD_fake - criticD_real + gradient_penalty
            optimizerD.step()
        
        netG.train()
        netD.eval()

        # Train G
        netG.zero_grad()

        noise.normal_(0, 1)
        fake = netG(noise, batch_att)
        criticG_fake = netD(fake, batch_att)
        criticG_fake = criticG_fake.mean()
        G_cost = -criticG_fake
        
        # Classifier loss
        c_errG = cls_criterion(pre_cla(fake), Variable(utils.map_label(batch_l, seenclasses)))
       
        
        # Bi-sematic reconstruct loss
        netRS.eval()
        netRU.eval()
        
        # Seen reconstruct loss
        syn_feature_r, syn_label_r, syn_att = generate_syn_feature_with_grad_and_att(netG, seenclasses, data.attribute, opts.syn_num_s)
        g_att = netRS(syn_feature_r)
        r_errGS = mse_criterion(g_att, syn_att)
        
        # Unseen reconstruct loss
        syn_feature_r, syn_label_r, syn_att = generate_syn_feature_with_grad_and_att(netG, unseenclasses, data.attribute, opts.syn_num)
        g_att = netRU(syn_feature_r)
        r_errGU = mse_criterion(g_att, syn_att)
        
        errG = G_cost + opts.cls_weight * c_errG + 1 * r_errGS + 1 * r_errGU
        errG.backward()
        optimizerG.step()
        

        # Train seen reconstructor
        netRS.train()
        netRU.train()
        netG.eval()
        syn_feature_r, syn_label_r, syn_att = generate_syn_feature_with_grad_and_att(netG, data.seenclasses, data.attribute, opts.syn_num)
        netRS.zero_grad()
        g_att = netRS(syn_feature_r)
        errRS = mse_criterion(g_att, syn_att)
        errRS.backward()
        optimizerRS.step()
        # train unseen reconstructor
        syn_feature_r, syn_label_r, syn_att = generate_syn_feature_with_grad_and_att(netG, data.unseenclasses, data.attribute, opts.syn_num)
        netRU.zero_grad()
        g_att = netRU(batch_feat)
        errRU = mse_criterion(g_att, batch_att)
        errRU.backward()
        optimizerRU.step()


    # Save and test
    logger.log('Epoch[%d/%d]:' % (epoch+1, opts.nepoch))
    torch.save(netG,model_save_path + "/" + "netG_" + str(epoch) + ".pt")
    torch.save(netD,model_save_path + "/" + "netD_" + str(epoch) + ".pt")
    torch.save(netRS,model_save_path + "/" + "netRS_" + str(epoch) + ".pt")
    torch.save(netRU,model_save_path + "/" + "netRU_" + str(epoch) + ".pt")
    netG.eval()
    netD.eval()
    netRS.eval()
    netRU.eval()
    netRS.to("cpu")
    netRU.to("cpu")
    
    # Generate unseen data
    syn_feature, syn_label, syn_att = generate_syn_feature_att(netG, data.unseenclasses, data.attribute, opts.cla_syn_num, device = opts.device)
    
    # combine sematic descriptions and attributes
    if opts.sematic_cla:
        syn_feature_att = torch.cat([syn_feature, syn_att], 1)
    else:
        syn_feature_att = syn_feature

    # Build test train dataset
    data_fa = argparse.Namespace()
    # Combie the reconstructed sematic descriptions and seen visual data 
    test_seen_att = opts.lambda2 * netRS(data.test_seen_feature)  + (1 - opts.lambda2) * netRU(data.test_seen_feature)
    if opts.sematic_cla:
        data_fa.test_seen_feature = torch.cat([data.test_seen_feature, test_seen_att], 1)
    else:
        data_fa.test_seen_feature = data.test_seen_feature
    
    data_fa.test_seen_label = data.test_seen_label 
    
    # Combie the reconstructed sematic descriptions and unseen visual data 
    test_unseen_att =  opts.lambda2 * netRS(data.test_unseen_feature)  + (1 - opts.lambda2) * netRU(data.test_unseen_feature)
    if opts.sematic_cla:
        data_fa.test_unseen_feature =  torch.cat([data.test_unseen_feature, test_unseen_att], 1)
    else:
        data_fa.test_unseen_feature =  data.test_unseen_feature
    data_fa.test_unseen_label = data.test_unseen_label
    data_fa.seenclasses = data.seenclasses
    data_fa.unseenclasses = data.unseenclasses
    
    if opts.sematic_cla:
        data_fa.train_feature = torch.cat([data.train_feature, data.attribute[data.train_label]], 1)
    else:
        data_fa.train_feature = data.train_feature
    
    data_fa.train_label = data.train_label
    data_train_feature = data_fa.train_feature
    data_train_label = data_fa.train_label
    

    # GZSL test
    if opts.gzsl:
        train_X = torch.cat((data_train_feature, syn_feature_att), 0)
        train_Y = torch.cat((data_train_label, syn_label), 0)
        nclass = opts.nclass_all
        cls = classifier.CLASSIFIER(train_X, train_Y, 
                         data_fa,
                         nclass,
                         True, 
                         opts.classifier_lr,
                         0.5,
                         opts.cla_epoch, 
                         opts.batch_size,#batch_size
                         True,
                         cla=opts.cla,
                         device=opts.device,
                         logger=logger)
        del cls, train_X, train_Y

    # ZSL test
    if opts.zsl:
        #     # zsl
        cls =  classifier.CLASSIFIER(syn_feature_att, 
                         utils.map_label(syn_label, data.unseenclasses), 
                         data_fa,     
                         data.unseenclasses.size(0), 
                         True,
                         opts.classifier_lr, 
                         0.5, 
                         opts.cla_epoch, 
                         opts.batch_size, #batch_size         
                         generalized=False, 
                         cla=opts.cla,
                        device=opts.device,
                        logger=logger)
        del cls
    logger.log('D:%.6f, G:%.6f, R1:%.6f, R2:%.6f' % (D_cost.cpu().detach().item(), errG.cpu().detach().item(), errRS.detach().item(), errRU.detach().item()))


        
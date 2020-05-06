#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 09:25:35 2019

@author: guixj
"""
""" copy from new_SN_CIFAR_10 and change to CIFAR_100 """

import numpy as np
import random
import torch
import torchvision
from torch.utils import data
from torchvision import transforms


class SN_cifar100_revised_trainset(data.Dataset):  
    """ symmetric noise train set """
    
    def __init__(self, raw_trainset, train_ind, flip_labels): 
        self.trainset = raw_trainset
        self.train_ind = train_ind
        self.flip_labels = flip_labels
        
    def __getitem__(self, index): # index in [0, 45000), not raw index
        
        feature, true_label = self.trainset[self.train_ind[index]]
        #flip_label = symmetric_flip(self.uniform_noise, true_label) #如此写代码是否 label 仍然未固定? deadly bug!!!
        flip_label = self.flip_labels[self.train_ind[index]]
        
        return feature, flip_label, index, true_label  # return index so to recode revised label

    def __len__(self):
        return len(self.train_ind)


class part_SN_cifar100_revised_trainset(data.Dataset):  
    """ symmetric noise train set """
    
    def __init__(self, raw_trainset, part_train_ind, noise_labels_50000): 
        self.trainset = raw_trainset
        self.part_train_ind = part_train_ind
        self.noise_labels = noise_labels_50000
        
    def __getitem__(self, index): # index in [0, 45000), not raw index
        
        feature, true_label = self.trainset[self.part_train_ind[index]]
        noise_label = self.noise_labels[self.part_train_ind[index]]
        
        return feature, int(noise_label), index, true_label  # return index so to recode revised label

    def __len__(self):
        return len(self.part_train_ind)


#class random_label_trainset(data.Dataset):  #understanding deep learning requires rethinking generalization
#    """ random label train set """
#    
#    def __init__(self, random_labels): 
#        self.trainset = torchvision.datasets.CIFAR100(
#            root = './data/',
#            train = True,
#            download = True,
#            transform = transforms.Compose([
#                transforms.ToTensor(),
#                transforms.Normalize(mean=[0.5071, 0.4865, 0.4465],
#                     std=[0.2673, 0.2564, 0.2762]) # from:gist.github.com/weiaicunzai
#    ])
#        )
#        self.random_labels = random_labels
#        
#    def __getitem__(self, index): # index in [0, 45000), but not raw index
#        
#        feature, true_label = self.trainset[index]
#        #random_label = np.random.choice(range(10))  #如此写代码是否 label 仍然未固定
#        random_label = self.random_labels[index]
#        return feature, random_label, index, true_label  # return index so to recode revised label
#
#    def __len__(self):
#        return len(self.trainset)

    
class cifar100_revised_valset(data.Dataset): # a small clean dataset
    """ a small clean valdation set """
    
    def __init__(self, val_ind): # only "val_ind" example used as test 

        self.trainset = \
            torchvision.datasets.CIFAR100(
                            root = './data/',
                            train = True,
                            download = True,
                            transform = transforms.Compose([
                        #        transforms.RandomCrop(32, 4), # 验证集和测试集一样，不应该做数据增强
                        #        transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5071, 0.4865, 0.4465],
                                     std=[0.2673, 0.2564, 0.2762]) # from:  https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151
                                ])
            )
                                
        from collections import defaultdict
        self.stratified_ind = defaultdict(list)
        # 分层采样抽 500 个
    
        for ind in val_ind:
            _, label = self.trainset[ind]
            self.stratified_ind[label].append(ind)  
            
        new_val_ind = []
        for key, value in self.stratified_ind.items():  # enumerate label1, label2, ...
            label_i_ind = value
            sel_ind = np.random.choice(label_i_ind, 50, replace=False) # those selected ind
            new_val_ind.extend(sel_ind)    
                
        self.ind = new_val_ind  # 500 = 50*10
        print("val size:", len(new_val_ind))
        
    def __getitem__(self, index):   
        example = self.trainset[self.ind[index]]
        return example

    def __len__(self):
        return len(self.ind) 


def split_raw_trainset(raw_trainset, noise): # a strafyied sampling
    
    raw_train_num  = len(raw_trainset)
    true_labels = np.zeros(raw_train_num, dtype=int)
    flip_labels = np.zeros(raw_train_num, dtype=int)
    val_num = int(raw_train_num*0.1)
    print("validtaion set num: ", val_num)
    
    from collections import defaultdict
    label_ind = defaultdict(list)
   
    for i, (_, label) in enumerate(raw_trainset):
        true_labels[i] = label
        label_ind[label].append(i)
        
        ### generate noise label
        if np.random.rand() > noise:
            flip_labels[i] = label
        else: # flip to any class
            flip_labels[i] = np.random.randint(0, 100)
    
    val_ind = []
    for label_class, part_ind in label_ind.items():  # enumerate label1, label2, ...
        label_i_ind = part_ind
        label_i_num = len(label_i_ind)
        sel_ind = np.random.choice(label_i_ind, np.int(0.1*label_i_num), replace=False) # those selected ind
        val_ind.extend(sel_ind)
    
    train_ind = list(set(range(raw_train_num)) - set(val_ind))
    
    # test it
    label_ind_count = np.zeros(100)
    for ind in val_ind:
        example, label = raw_trainset[ind]
        label_ind_count[label] += 1
    assert np.all(label_ind_count == np.ones(100)*50)
    print("all class in validation set are 50")
    
    return train_ind, val_ind, true_labels, flip_labels


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataset(noise_rate):

    # np.random.seed(10)
    #seed_torch(20)
    
    seed_torch(10)  #9月6日
    
    normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4465],
                                     std=[0.2673, 0.2564, 0.2762]) # from:gist.github.com/weiaicunzai
    
    raw_trainset = torchvision.datasets.CIFAR100(
        root = './data/',
        train = True,
        download = True,
        transform = transforms.Compose([
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    
#    raw_trainset_test = torchvision.datasets.CIFAR100(
#        root = './data/',
#        train = True,
#        download = True,
#        transform = transforms.Compose([ # not use RandomCrop, RandomHorizontalFlip
#            transforms.ToTensor(),
#            normalize,
#        ]))

    raw_testset = torchvision.datasets.CIFAR100(
        root = './data/',
        train = False,
        download = True,
        transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
        ]))
    
    raw_trainloader = torch.utils.data.DataLoader(
        raw_trainset,
        batch_size = 128,
        shuffle = True,
        )
    
    raw_testloader = torch.utils.data.DataLoader(
        raw_testset,
        batch_size = 128,
        shuffle = True,
        )
    
    train_ind, val_ind, true_labels, flip_labels = split_raw_trainset(raw_trainset, noise_rate)

    my_trainloader = torch.utils.data.DataLoader(
        SN_cifar100_revised_trainset(raw_trainset, train_ind, flip_labels),
        batch_size = 128,
        shuffle = True,
        )
    
#    my_train_loader_test = torch.utils.data.DataLoader(
#        SN_cifar100_revised_trainset(raw_trainset_test, train_ind, flip_labels),
#        batch_size = 128,
#        shuffle = True,
#        )
    
    
#    random_labels = np.random.choice(range(10), size=50000) # generate random labels
    
#    random_trainloader = torch.utils.data.DataLoader(
#        random_label_trainset(random_labels),
#        batch_size = 128,
#        shuffle = True,
#        )
    
    my_valloader = torch.utils.data.DataLoader(
        cifar100_revised_valset(val_ind),
        batch_size = 128,
        shuffle = True,
        )

    return raw_trainloader, raw_testloader, my_trainloader, \
                my_valloader, train_ind, flip_labels 


def get_part_data(part_train_ind, label_50000, need_test_loader=False):
    
    seed_torch(10)
    normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4465],
                                     std=[0.2673, 0.2564, 0.2762]) # from:gist.github.com/weiaicunzai

    raw_trainset = torchvision.datasets.CIFAR100(
        root = './data/',
        train = True,
        download = True,
        transform = transforms.Compose([
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
        
    part_trainloader = torch.utils.data.DataLoader(
        part_SN_cifar100_revised_trainset(raw_trainset, part_train_ind, label_50000),
        batch_size = 128,
        shuffle = True,
        )
    
    if not need_test_loader:
        return part_trainloader
    
    else:
        raw_trainset_test = torchvision.datasets.CIFAR100(
            root = './data/',
            train = True,
            download = True,
            transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
    
        part_trainloader_test = torch.utils.data.DataLoader(
            part_SN_cifar100_revised_trainset(raw_trainset_test, part_train_ind, label_50000),
            batch_size = 128,
            shuffle = True,
            )
        
        return part_trainloader, part_trainloader_test

 
if __name__ == "__main__":

    noise = 0.7
    raw_trainloader, raw_testloader, my_trainloader, my_valloader, *rest  = get_dataset(noise_rate=noise)
    
    
    for epoch in range(2):
        count_y = np.zeros(10)
        for x_batch, y_batch, *rest in my_trainloader:
            for t in range(10):
                count_y[t] += np.sum(y_batch.numpy() == t)
        print(count_y)
    
#    [4532. 4468. 4526. 4370. 4542. 4498. 4465. 4518. 4593. 4488.]
#    [4533. 4437. 4501. 4549. 4544. 4502. 4455. 4454. 4558. 4467.]
#    # save data
#    import pickle # pickle 并没有编码 定义的类，需要辅助信息才行
#    with open(r'./SNdata7_29_noise%.2f.pkl'%noise, 'wb') as inp:
#        pickle.dump(my_trainloader, inp)
#        pickle.dump(my_valloader, inp)
#        pickle.dump(raw_testloader, inp)
#        pickle.dump(raw_trainloader, inp)







#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Structured label noise CIFAR-10 """

import numpy as np
import random
import torch
import torchvision
from torch.utils import data
from torchvision import transforms


class ST_cifar10_revised_trainset(data.Dataset):  
    """ structured label noise train set """
    
    def __init__(self, raw_trainset, train_ind, flip_labels): 
        self.trainset = raw_trainset
        self.train_ind = train_ind
        self.flip_labels = flip_labels
        
    def __getitem__(self, index): # index in [0, 45000), not raw index
        
        feature, true_label = self.trainset[self.train_ind[index]]
        flip_label = self.flip_labels[self.train_ind[index]]
        
        return feature, flip_label, index, true_label  # return index so to recode revised label

    def __len__(self):
        return len(self.train_ind)


class part_ST_cifar10_revised_trainset(data.Dataset):  # use it to get selected data

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

class cifar10_revised_valset(data.Dataset):
    """ clean valdation set """
    
    def __init__(self, val_ind): 

        self.trainset = torchvision.datasets.CIFAR10(
            root = './data/',
            train = True,
            download = True,
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.491, 0.482, 0.447],
                     std=[0.247, 0.243, 0.262])
                        ])
                    )   
                
        self.ind = val_ind 
        
    def __getitem__(self, index):
        return self.trainset[self.ind[index]]

    def __len__(self):
        return len(self.ind) 


def ST_split_raw_trainset(raw_trainset, noise): # a strafyied sampling
    
    raw_train_num  = len(raw_trainset)
    true_labels = np.zeros(raw_train_num, dtype=int)
    flip_labels = np.zeros(raw_train_num, dtype=int)
    val_num = int(raw_train_num*0.1)
    print("validtaion set num: ", val_num)
    
    from collections import defaultdict
    label_ind = defaultdict(list)
   
    """ structured label noise
    0-ariplane, 1-automobile, 2-bird, 3-cat, 4-deer,
    5-dog, 6-frog, 7-horse, 8-ship, 9-truck  # from pytorch.org
    so: 9->1, 2->0, 4->7, 3->5, 5->3
    """
    noisy_label = [0, 1, 0, 5, 7, 3, 6, 7, 8, 1]
    
    for i, (_, label) in enumerate(raw_trainset):
        true_labels[i] = label
        label_ind[label].append(i)

        ### generate noise label
        if np.random.rand() > noise:
            flip_labels[i] = label
        else: # filp to the corresponding class
            flip_labels[i] = noisy_label[label]
    
    val_ind = []
    for label_class, part_ind in label_ind.items():  # enumerate label1, label2, ...
        label_i_ind = part_ind
        label_i_num = len(label_i_ind)
        sel_ind = np.random.choice(label_i_ind, np.int(0.1*label_i_num), replace=False) # those selected ind
        val_ind.extend(sel_ind)
    
    train_ind = list(set(range(raw_train_num)) - set(val_ind))
    
    # test it
    label_ind_count = np.zeros(10)
    for ind in val_ind:
        example, label = raw_trainset[ind]
        label_ind_count[label] += 1
    assert np.all(label_ind_count == np.ones(10)*500)
    print("all class in validation set are 500")
    
    return train_ind, val_ind, true_labels, flip_labels


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataset(noise_rate):

    seed_torch(0)
    
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447],
                                     std=[0.247, 0.243, 0.262]) 
    
    raw_trainset = torchvision.datasets.CIFAR10(
        root = './data/',
        train = True,
        download = True,
        transform = transforms.Compose([
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    

    raw_testset = torchvision.datasets.CIFAR10(
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
    
    train_ind, val_ind, true_labels, flip_labels = ST_split_raw_trainset(raw_trainset, noise_rate)

    my_trainloader = torch.utils.data.DataLoader(
        ST_cifar10_revised_trainset(raw_trainset, train_ind, flip_labels),
        batch_size = 128,
        shuffle = True,
        )

    my_valloader = torch.utils.data.DataLoader(
        cifar10_revised_valset(val_ind),
        batch_size = 128,
        shuffle = True,
        )

    return raw_trainloader, raw_testloader, my_trainloader, \
                my_valloader, train_ind, flip_labels 


def get_part_data(part_train_ind, label_50000, need_test_loader=False):
    
    seed_torch(20)
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447],
                                     std=[0.247, 0.243, 0.262]) 

    raw_trainset = torchvision.datasets.CIFAR10(
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
        part_ST_cifar10_revised_trainset(raw_trainset, part_train_ind, label_50000),
        batch_size = 128,
        shuffle = True,
        )
    
    if not need_test_loader:
        return part_trainloader
    
    else:
        raw_trainset_test = torchvision.datasets.CIFAR10(
            root = './data/',
            train = True,
            download = True,
            transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
    
        part_trainloader_test = torch.utils.data.DataLoader(
            part_ST_cifar10_revised_trainset(raw_trainset_test, part_train_ind, label_50000),
            batch_size = 128,
            shuffle = True,
            )
        
        return part_trainloader, part_trainloader_test


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt


def epoch_train(net, optimizer, train_loader, device='cuda'): # stage 1, training on noise labels
    
    net.train()        
    for x_batch, y_batch, *rest in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        pred = net(x_batch)
        loss = F.cross_entropy(pred, y_batch)

        #print('loss:', loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    
def epoch_test(net, test_loader, device='cuda'):
    
    net.eval() 
    correct = 0
    num_sum = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            num_sum += len(y_batch)
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            output = net(x_batch)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(y_batch.view_as(pred)).sum().item()
    
    return correct / num_sum  


def return_noisy_true_label(train_loader):
    
    num = np.sum([len(x) for x, *rest in train_loader])
    noise_labels = np.ones(num, dtype=int)*(-1)
    true_labels = np.ones(num, dtype=int)*(-1)
    
    for x_batch, y_batch, index_batch, true_y_batch in train_loader:
            
        noise_labels[index_batch.numpy()] = y_batch.cpu().numpy()
        true_labels[index_batch.numpy()] = true_y_batch.cpu().numpy()
    
    return noise_labels, true_labels


def epoch_count_entropy_loss(net, train_loader, device='cuda', class_num=10):
    # count loss, entropy in train_loader
    net.eval()
    num = np.sum([len(x) for x, *rest in train_loader])
    noise_labels = np.ones(num, dtype=int)*(-1)
    true_labels = np.ones(num, dtype=int)*(-1)
    pred_labels = np.ones(num, dtype=int)*(-1)
    
    all_loss = np.zeros([num])
    all_entropy = np.zeros([num])
    all_softmax = np.zeros([num, class_num])
    
    with torch.no_grad():
        for x_batch, y_batch, index_batch, true_y_batch in train_loader:
            
            noise_labels[index_batch.numpy()] = y_batch.cpu().numpy()
            true_labels[index_batch.numpy()] = true_y_batch.cpu().numpy()
            
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            true_y_batch = true_y_batch.to(device)
            
            softmax_output = F.softmax(net(x_batch), dim=1)
            all_softmax[index_batch.numpy(),:] = softmax_output.detach().cpu().numpy()
            
            pred = torch.argmax(softmax_output, dim=1)
            pred_labels[index_batch.numpy()] = pred.detach().cpu().numpy()

            loss_batch = -1 * torch.log(softmax_output[np.arange(softmax_output.size(0)), y_batch.squeeze()])
            
            all_loss[index_batch.numpy()]= loss_batch.detach().cpu().squeeze().numpy()
            
            entropy_batch = -torch.sum(softmax_output*torch.log(softmax_output), dim=1)
            all_entropy[index_batch.numpy()]= entropy_batch.detach().cpu().squeeze().numpy()
        
    return noise_labels, true_labels, pred_labels, all_loss, all_entropy, all_softmax


def epoch_test_train_loader(net, train_loader, Founder="record_y", name=None, device='cuda', low_threshold=0.5): 

    train_size = np.sum((len(t) for t, *rest in train_loader)) 
    model_predict_y = np.ones(train_size)*(-1)
    
    
    net.eval() 
    true_correct = 0
    noise_correct = 0
    noise_correct1 = 0
    noise_correct2 = 0
    num_sum = 0
    
    with torch.no_grad():
        count_low = 0
        count_error = 0
        count_error_low = 0
        confusion_matrix_train = np.zeros([100, 100])

        for x_batch, y_batch, index_batch, true_y_batch in train_loader:
            
            num_sum += len(true_y_batch)
            x_batch = x_batch.to(device)
            
            y_batch = y_batch.to(device)
            true_y_batch = true_y_batch.to(device)
            output = net(x_batch)
            pred = output.max(1, keepdim=True)[1]
            #_, pred = torch.max(outputs.data, 1)
            true_correct += pred.eq(true_y_batch.view_as(pred)).sum().item()
            
            A = pred.detach().cpu().numpy().reshape(-1,1) 
            B = y_batch.cpu().numpy().reshape(-1,1)
            C = true_y_batch.cpu().numpy().reshape(-1,1)
            
            noise_correct1 += np.sum((A == B) & (B == C))
            noise_correct2 += np.sum((A == B) & (B != C))
            
            #print(noise_correct1)
            #print(noise_correct2)
            noise_correct += pred.eq(y_batch.view_as(pred)).sum().item()
            #print(noise_correct)
            assert noise_correct1 + noise_correct2 == noise_correct
            
            model_predict_y[index_batch.numpy()] = pred.detach().cpu().squeeze().numpy()
            
            for x, y, z in zip(true_y_batch.cpu().numpy(), pred.detach().cpu().numpy(), F.softmax(output,dim=1).detach().cpu().numpy()):
                
                if np.max(z)< low_threshold:
                        count_low += 1  
                if x != y:
                    count_error += 1
                    #print(np.max(z))
                    if np.max(z)<  low_threshold:
                        count_error_low += 1
                    confusion_matrix_train[x, y] += 1
        rate_confusion_matrix_train = confusion_matrix_train/np.sum(confusion_matrix_train, axis=1).reshape(100, 1)
            
    if name != None: # if name == None, then not record
        model_predict_y_file = Founder + name + "_y"
        np.save(model_predict_y_file, model_predict_y)
        model_predict_y_d = np.zeros([45000, 10]) # datanum, classnum
        for i, label in enumerate(model_predict_y):
            model_predict_y_d[i, int(label)] = 4.5
        model_predict_y_d = np.exp(model_predict_y_d) / np.sum(model_predict_y_d, axis=1).reshape(-1, 1)
        model_predict_y_d_file = Founder + name+"_y_d"
        np.save(model_predict_y_d_file, model_predict_y_d)

    return true_correct / num_sum, noise_correct/num_sum, noise_correct1/num_sum, noise_correct2/num_sum, num_sum, count_low, count_error, count_error_low


class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, title=None, resume=False): 
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            if resume: 
                self.file = open(fpath, 'r') 
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')  
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume: 
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()

    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def plot(self, names=None):   
        names = self.names if names == None else names
        numbers = self.numbers
        for _, name in enumerate(names):
            x = np.arange(len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]))
        plt.legend([self.title + '(' + name + ')' for name in names])
        plt.grid(True)

    def close(self):
        if self.file is not None:
            self.file.close()


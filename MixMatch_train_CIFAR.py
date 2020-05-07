
# for CIFAR-10 
# preact_resnet32
# alpha = 0.75 lambda_u = 75, following MixMatch NeurIPS-19 paper


# for CIFAR-100
# resnet-34
# alpha = 0.75 lambda = 150, following MixMatch NeurIPS-19 paper


from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler

import network_models.preact_resnet as preact_resnet
import network_models.resnet as resnet
import MixMatch_dataset.noise_data_augument_CIFAR10 as CIFAR_10_dataset
import MixMatch_dataset.noise_data_augument_CIFAR100 as CIFAR_100_dataset
from MixMatch_dataset.noise_data_augument_CIFAR10 import get_cifar10_select_ind
from MixMatch_dataset.noise_data_augument_CIFAR10 import get_cifar100_select_ind
from MixMatch_utils import Bar, Logger, AverageMeter, accuracy, mkdir_p 

#from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='Pytorch MixMatch Training')
# Optimization options
parser.add_argument('--epochs', default=1024, type=int, metavar='N', # 1024
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,  # 0.002 -> 0.001
                    metavar='LR', help='initial learning rate')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
#Device options
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
#Method options
parser.add_argument('--n-labeled', type=int, default=4000,
                        help='Number of labeled data')
parser.add_argument('--val-iteration', type=int, default=1024,
                        help='Number of labeled data')
parser.add_argument('--out', default='MixMatch_result',
                        help='Directory to output the result')
parser.add_argument('--alpha', default=0.75, type=float)
parser.add_argument('--lambda-u', default=75, type=float)  # 75 -> 150
parser.add_argument('--T', default=0.5, type=float)
parser.add_argument('--ema-decay', default=0.999, type=float)


#args = parser.parse_args()
args = parser.parse_args(['--epochs', '1024', '--start-epoch', '0', '--batch-size', '64', '--lr', '0.001'])

state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)

best_test_acc = 0  # best test accuracy
best_acc = 0


def Weighted_MixMatch_main(all_data_information, dataset_name, noise_rate, select_num):  # dataset_name can be "xxxCIFAR10" or ""xxxCIFAR100"
    global best_acc
    global best_test_acc 

    if not os.path.isdir(args.out):
        mkdir_p(args.out)
    
    if dataset_name.endswith('CIFAR10'):
        dataset = CIFAR_10_dataset
        get_data = get_cifar10_select_ind
        net = preact_resnet.preact_resnet32
        args.lambda_u = 75 
        print(f'==> Preparing cifar10')

    elif dataset_name.endswith("CIFAR-100"):
        dataset = CIFAR_100_dataset
        get_data = get_cifar100_select_ind
        net = resnet.resnet_34
        args.lambda_u = 150  # from 75 -> 150
        print(f'==> Preparing cifar100')

    else:
        raise ValueError('Error: no appropriate dataset is given')
        
    # Data
    transform_train = transforms.Compose([
        dataset.RandomPadandCrop(32),
        dataset.RandomFlip(),
        dataset.ToTensor(),
    ])

    transform_val = transforms.Compose([
        dataset.ToTensor(),
    ])
    
    train_labeled_set, weight_list, train_unlabeled_set, val_set, test_set = get_data(all_data_information, transform_train=transform_train, transform_val=transform_val)
    
    sampler = WeightedRandomSampler(  # weighted resampling
        weights=weight_list,
        num_samples=len(weight_list),
        replacement=True
        )
    
    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, sampler=sampler,shuffle=True, num_workers=0, drop_last=True) # drop last
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    if dataset_name.endswith('CIFAR10'):
        print("==> creating preact_resnet_32")
    elif dataset_name.endswith('CIFAR100'):
        print("==> creating resnet_34")
        
    def create_model(ema=False):
        model = net()
        model = model.cuda()

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model()
    ema_model = create_model(ema=True) # Exponential moving average model(decay rate = 0.999)

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    train_criterion = SemiLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    ema_optimizer= WeightEMA(model, ema_model, net, alpha=args.ema_decay)
    start_epoch = 0

    name = "MixMatch_log_for_%s_%.1f_%d"%(dataset_name, noise_rate, select_num)+".txt"
    # Resume
    title = dataset_name
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.resume, name), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.out, name), title=title)
        logger.set_names(['epoch', 'Train Loss', 'Train Loss X', 'Train Loss U',  'Valid Loss', 'Valid Acc.', 'Test Loss', 'Test Acc.'])

    #writer = SummaryWriter(args.out)
    #step = 0
    test_accs = []
    # Train and val
    for epoch in range(start_epoch, args.epochs):

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_loss_x, train_loss_u = train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, train_criterion, epoch, use_cuda)
        _, train_acc = validate(labeled_trainloader, ema_model, criterion, epoch, use_cuda, mode='Train Stats')
        val_loss, val_acc = validate(val_loader, ema_model, criterion, epoch, use_cuda, mode='Valid Stats')
        test_loss, test_acc = validate(test_loader, ema_model, criterion, epoch, use_cuda, mode='Test Stats ')

        #step = args.batch_size * args.val_iteration * (epoch + 1)

        # append logger file
        logger.append([int(epoch), train_loss, train_loss_x, train_loss_u, val_loss, val_acc, test_loss, test_acc])

        # save model
        is_best = val_acc > best_acc
        if is_best:
            best_test_acc = test_acc 
            best_acc = max(val_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'acc': val_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best)
        test_accs.append(test_acc)
    logger.close()


def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion, epoch, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    ws = AverageMeter()
    end = time.time()

    bar = Bar('Training', max=args.val_iteration)
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    
    model.train()
    for batch_idx in range(args.val_iteration):  # val_iteration
        try:
            inputs_x, targets_x = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader) # iterate train_loader, get a minibatch
            inputs_x, targets_x = labeled_train_iter.next()

        try:
            (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()

        # measure data loading time
        data_time.update(time.time() - end)

        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        targets_x = torch.zeros(batch_size, 10).scatter_(1, targets_x.view(-1,1), 1)

        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u = inputs_u.cuda()
            inputs_u2 = inputs_u2.cuda()


        with torch.no_grad():
            # compute guessed labels of unlabel samples
            outputs_u = model(inputs_u)
            outputs_u2 = model(inputs_u2)
            p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
            pt = p**(1/args.T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        # mixup
        all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)

        l = np.random.beta(args.alpha, args.alpha)

        l = max(l, 1-l)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        # interleave labeled and unlabed samples between batches to get correct batchnorm calculation 
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        logits = [model(mixed_input[0])]
        for input in mixed_input[1:]:
            logits.append(model(input))

        # put interleaved samples back
        logits = interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)

        Lx, Lu, w = criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:], epoch+batch_idx/args.val_iteration)

        loss = Lx + w * Lu

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))
        ws.update(w, inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f} | W: {w:.4f}'.format(
                    batch=batch_idx + 1,
                    size=args.val_iteration,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    w=ws.avg,
                    )
        bar.next()
    bar.finish()

    ema_optimizer.step(bn=True)

    return (losses.avg, losses_x.avg, losses_u.avg,)


def validate(valloader, model, criterion, epoch, use_cuda, mode):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            
            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(valloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
        bar.finish()
    return (losses.avg, top1.avg)


def save_checkpoint(state, is_best, checkpoint=args.out, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, args.lambda_u * linear_rampup(epoch) # change w in here


class WeightEMA(object):
    def __init__(self, model, ema_model, net, alpha=0.999): # weight exponential moving average
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.tmp_model = net().cuda()  # use net 
        self.wd = 0.02 * args.lr  # revise weight decay size

        for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
            ema_param.data.copy_(param.data)

    def step(self, bn=False):
        if bn:
            # copy batchnorm stats to ema model
            for ema_param, tmp_param in zip(self.ema_model.parameters(), self.tmp_model.parameters()):
                tmp_param.data.copy_(ema_param.data.detach())

            self.ema_model.load_state_dict(self.model.state_dict())

            for ema_param, tmp_param in zip(self.ema_model.parameters(), self.tmp_model.parameters()):
                ema_param.data.copy_(tmp_param.data.detach())
        else:
            one_minus_alpha = 1.0 - self.alpha
            for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                ema_param.data.mul_(self.alpha)
                ema_param.data.add_(param.data.detach() * one_minus_alpha)
                # customized weight decay
                param.data.mul_(1 - self.wd)


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


if __name__ == '__main__':
    pass
    #MixMatch_main(all_data_name, dataset_name, noise_rate, select_num)

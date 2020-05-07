""" Step 1: CE (cross entropy) training and recording loss values """
""" Step 2: Selecting small-loss data """
""" Step 3: Ablation study --- CE training only on selected data (This step can be omitted) """
""" Step 4: Using Weighted MixMatch """

import os
import numpy as np
import torch 
import torch.optim as optim
from util import epoch_train, epoch_test, epoch_count_entropy_loss, Logger    
from util import return_noisy_true_label
from network_models.preact_resnet import preact_resnet32
from network_models.resnet import ResNet34

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-r', '--noise_rate', type=float)
parser.add_argument('-d', '--dataset', 
                    help="Dataset: 'Uniform_CIFAR10', 'Pairwise_CIFAR10', 'Structured_CIFAR10', 'Uniform_CIFAR100', 'Pairwise_CIFAR100'")
parser.add_argument('-e', '--sel_epoch', default=70, type=int, 
                    help='[0, sel_epoch] will be used to calculated mean loss')
parser.add_argument('-b', '--beta', default=0.2, type=float, 
                    help='scale parameter used to select less data than (1-\eta_i) ')
parser.add_argument('-g', '--gamma_index', default=1, type=int, 
                    help='gamma can be chosen from [1, (1+z)/2, z], gamma_index=1 corresponding (1+z)/2')
parser.add_argument('-k', '--kappa', default=-np.log(0.7), type=float, 
                    help='the parameter \kappa for weighting ')

args = parser.parse_args()

#args = parser.parse_args(['-d', 'AN_CIFAR10', '-r', '0.4'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.dataset == 'Uniform_CIFAR10':
    from noisy_data_generator.Uniform_CIFAR10 import get_dataset, get_part_data
    true_noise_rate = args.noise_rate*0.9
    
elif args.dataset == 'Pairwise_CIFAR10':
    from noisy_data_generator.Pairwise_CIFAR10 import get_dataset, get_part_data
    true_noise_rate = args.noise_rate
    
elif args.dataset == 'Structured_CIFAR10':
    from noisy_data_generator.Structured_CIFAR10 import get_dataset, get_part_data
    true_noise_rate = args.noise_rate*0.5
    each_class_noise_rate = [args.noise_rate/(1+args.noise_rate), args.noise_rate/(1+args.noise_rate),\
                             0, args.noise_rate, 0, args.noise_rate, 0, args.noise_rate/(1+args.noise_rate), 0, 0]
    
elif args.dataset == 'Uniform_CIFAR100':
    from noisy_data_generator.Uniform_CIFAR100 import get_dataset, get_part_data
    true_noise_rate = args.noise_rate*0.99

elif args.dataset == 'Pairwise_CIFAR100':
    from noisy_data_generator.Pairwise_CIFAR100 import get_dataset, get_part_data
    true_noise_rate = args.noise_rate

else:
    raise ValueError("Error: no appropriate dataset is given")


################ Step 1: CE (cross entropy) training and recording loss values ######################

if args.dataset.endswith('CIFAR10'):
    num_class = 10
    true_class_prior = [1.0/num_class]*num_class # use it to keep class balance for sample selection
    raw_model = preact_resnet32().to(device)
    optimizer = optim.SGD(raw_model.parameters(), lr=0.2,
                           momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,80])
    epoch_num = 120
    
elif args.dataset.endswith('CIFAR100'):
    num_class = 100
    true_class_prior = [1.0/num_class]*num_class
    raw_model = ResNet34().to(device)
    optimizer = optim.SGD(raw_model.parameters(), lr=0.01,
                           momentum=0.9, weight_decay=1e-4)
    
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,80])
    epoch_num = 120

raw_trainloader, raw_testloader, noisy_trainloader, val_loader, \
    train_ind, noisy_labels = get_dataset(noise_rate=args.noise_rate)

if not os.path.exists('CE_reproduce_result'):# reproduce the result of cross-entropy (directly training on noisy data)
    os.mkdir('CE_reproduce_result')

logger = Logger('CE_reproduce_result/log_for_%s_%.1f.txt'%(args.dataset, args.noise_rate))
logger.set_names(['epoch', 'val_accu', 'test_accu'])

val_best = 0.0
val_accu = []
best_epoch = 0   # the epoch num when val accu is best
test_best = 0.0  # the test accu when val accu is best
test_accu = []

for epoch in range(0, epoch_num): 
    epoch_train(raw_model, optimizer, noisy_trainloader)
    lr_scheduler.step()
    val_accuracy = epoch_test(raw_model, val_loader)
    val_accu.append(val_accuracy)
    test_accuracy = epoch_test(raw_model, raw_testloader)
    test_accu.append(test_accuracy)
    
    logger.append([epoch, val_accuracy, test_accuracy])
    print('epoch%d  val accu, test accu:'%(epoch), val_accuracy, test_accuracy)
    
    if val_accuracy >= val_best:
        val_best = val_accuracy
        best_epoch = epoch
        test_best = test_accuracy
    
    # calculate the loss values of each training example of the current model
    noisy_labels, true_labels, pred_labels, all_loss, all_entropy, all_softmax \
        = epoch_count_entropy_loss(raw_model, noisy_trainloader, class_num=num_class)
    
    # record each examples' loss
    if not os.path.exists("loss_record_result"):
        os.mkdir("loss_record_result")    
    folder = 'loss_record_result/%s_%.1f_loss_record'%(args.dataset, args.noise_rate)
    if not os.path.exists(folder):
        os.mkdir(folder)
    np.save(folder+"/all_loss_%d.npy"%epoch, all_loss)

logger.close()    

print('best_val epoch: %d'%best_epoch, test_best, 'last', test_accuracy, \
      'potential_best_test', np.max(test_accu))

with open("CE_reproduce_result/%s_%.1f_print_record.txt"%(args.dataset, args.noise_rate), 'w') as f:
    line1 = 'best_val epoch: %d, %.4f, last, %.4f'%(best_epoch, test_best, test_accuracy)
    line2 = 'potential_best_test: %.4f'%np.max(test_accu)
    f.writelines(line1 + "\n" + line2 + "\n")
    

# record the synthesized noisy data 
import pickle 
with open(r'CE_reproduce_result/%s_%.1f_synthesized_noisy_data'%(args.dataset, args.noise_rate), 'wb') as inp:
    pickle.dump(train_ind, inp)
    pickle.dump(noisy_labels, inp)
    pickle.dump(true_labels, inp)



################# Step 2: Selecting small-loss data ###########################

noisy_labels, true_labels = return_noisy_true_label(noisy_trainloader) 

def class_balance_num(true_class_prior, noisy_labels, num_class, prop): 
    
    num_class_noisy_label = [np.sum(noisy_labels == i) for i in range(num_class)]
    availiable_num = [num_class_noisy_label[i]*prop[i] for i in range(num_class)]
    gamma0, gamma2 = 1, max(availiable_num/true_class_prior[i])/min(availiable_num/true_class_prior[i]) # gamma = 1 or max(prop(i)n_i/p_i) / min(prop(i)n_i/p_i)
    gamma1 = (gamma0 + gamma2)/2   
    args.gamma = [gamma0, gamma1, gamma2][args.gamma_index] # defined as args.gamma
    val = args.gamma*np.min([availiable_num[i]/true_class_prior[i] for i in range(len(true_class_prior))]) # val s.t. val*p[i] <= gamma*availabel[i] for all i
    class_balance_num = [min(val*true_class_prior[i], availiable_num[i])for i in range(num_class)]
    
    return class_balance_num


if args.dataset == 'Structured_CIFAR10':
    prop = [max([1-(1+args.beta)*each_class_noise_rate[i], (1-args.beta)*(1-each_class_noise_rate[i])]) for i in range(10)] # proportion
    each_class_num = class_balance_num(true_class_prior, noisy_labels, num_class, prop)
else:
    prop = [max([1-(1+args.beta)*true_noise_rate, (1-args.beta)*(1-true_noise_rate)])]*10
    each_class_num = int(45000*prop/10)  # 这里是否也用 class_balanced_num

all_mean_loss = np.zeros(45000)
for e in range(args.sel_epoch): # [0, args.sel_epoch)
    loss_record = np.load("loss_record_result/%s_%.1f_loss_record/all_loss_%d.npy"%(args.dataset, args.noise_rate, e))
    all_mean_loss += loss_record
all_mean_loss /= args.sel_epoch
sort_ind = np.argsort(all_mean_loss)  # ranking loss


select_ind = []  # recording select index for each class
w_list = []      # calculating weights

for i in range(num_class): # filter class by class
    ind_i = sort_ind[np.where(noisy_labels[sort_ind] == i)[0]]
    sel_num_i = each_class_num[i]
    select_ind_i = ind_i[:int(sel_num_i)]
    select_ind.extend(select_ind_i)
    min_loss_i, max_loss_i = all_mean_loss[ind_i[0]], all_mean_loss[ind_i[-1]]
    t_list = [(all_mean_loss[j]-min_loss_i) / (max_loss_i-min_loss_i) for j in select_ind_i]
    w_list.extend([np.exp(-args.kappa*t) for t in t_list])


select_num = len(select_ind)
select_accuracy = np.mean(noisy_labels[select_ind] == true_labels[select_ind])
each_class_num = [np.sum(noisy_labels[select_ind] == i) for i in range(num_class)]
each_class_accu = np.zeros(num_class)
for i in range(num_class):
    select_ind_i  = np.where(noisy_labels[select_ind]==i)[0]
    each_class_accu[i] = np.mean(noisy_labels[select_ind][select_ind_i] == true_labels[select_ind][select_ind_i])

if not os.path.exists('selected_noisy_data_result'):
    os.mkdir('selected_noisy_data_result')
folder = 'selected_noisy_data_result/%s_%.1f'%(args.dataset, args.noise_rate)
if not os.path.exists(folder):
    os.mkdir(folder)

with open(folder+"/each_calss_num_accu_record.txt", 'a') as f:
    f.writelines("\n")
    line = "select_num: %d"%select_num + " selected epoch : %d"%args.sel_epoch + " accuracy:%.9f"%select_accuracy +"\n"
    f.writelines(line)
    line = "each class num: " + " ".join(str(x) for x in each_class_num) + "\n"
    f.writelines(line)
    line = "each class accu: " + " ".join(str(x) for x in each_class_accu) +"\n"
    f.writelines(line)

select_mask = np.zeros(45000) > 1
select_mask[select_ind] = True
raw_train_mask = np.zeros(50000) >1
raw_train_mask[train_ind] = True

select_train_ind = np.array(train_ind)[select_ind]  # taking as labeled data
unlabeled_train_ind =  np.array(train_ind)[~select_mask] # taking as unlabeled data
val_ind = np.array(range(50000))[~raw_train_mask]  # the val data

noisy_label50000 = np.ones(50000)*-1
noisy_label50000[select_train_ind] = noisy_labels[select_ind]  # only give noisy labels for labeled data 

np.save(folder+"/%d_labeled_ind_%.1f_sel_epoch_%d.npy"%(select_num, args.noise_rate, args.sel_epoch), select_train_ind)
np.save(folder+"/%d_w_list_%.1f_sel_epoch_%d.npy"%(select_num, args.noise_rate, args.sel_epoch), w_list)
np.save(folder+"/%d_unlabeled_ind_%.1f_sel_epoch_%d.npy"%(select_num, args.noise_rate, args.sel_epoch), unlabeled_train_ind)
np.save(folder+"/%d_val_ind_%.1f_sel_epoch_%d.npy"%(select_num, args.noise_rate, args.sel_epoch), val_ind)
np.save(folder+"/%d_noisy_label_50000_%.1f_sel_epoch_%d.npy"%(select_num, args.noise_rate, args.sel_epoch), noisy_label50000)


######################## Step 3: Ablation study --- CE training only on selected data (This step can be omitted) ################
    
part_train_ind = np.load(folder+"/%d_labeled_ind_%.1f_sel_epoch_%d.npy"%(select_num, args.noise_rate, args.sel_epoch))
label_50000 = np.load(folder+"/%d_noisy_label_50000_%.1f_sel_epoch_%d.npy"%(select_num, args.noise_rate, args.sel_epoch))
part_trainloader = get_part_data(part_train_ind, label_50000)

if args.dataset.endswith('CIFAR10'):
    num_class = 10
    part_model = preact_resnet32().to(device)
    optimizer = optim.SGD(part_model.parameters(), lr=0.2,
                          momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,80])
    epoch_num = 120
    
elif args.dataset.endswith('CIFAR100'):
    num_class = 100
    part_model = ResNet34(100).to(device)
    optimizer = optim.SGD(part_model.parameters(), lr=0.01,
                          momentum=0.9, weight_decay=1e-4)    
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,80])
    epoch_num = 120
    
if not os.path.exists('Ablation_study'):
    os.mkdir('Ablation_study')
    
logger = Logger('Ablation_study/Ablation_study-log_for_%s_%.1f_%d_sel_epoch_%d.txt'%(args.dataset, args.noise_rate, len(part_train_ind), args.sel_epoch))
logger.set_names(['epoch', 'val_accu', 'test_accu'])

val_accu = [] # recode val accu
val_best = 0.0
test_accu = []
test_best = 0.0  # test accu when val accu get best

for epoch in range(0, epoch_num): 
    epoch_train(part_model, optimizer, part_trainloader)
    lr_scheduler.step()
    val_accuracy = epoch_test(part_model, val_loader)
    val_accu.append(val_accuracy)
    test_accuracy = epoch_test(part_model, raw_testloader)
    test_accu.append(test_accuracy)
    
    if val_accuracy >= val_best:
        val_best = val_accuracy
        test_best = test_accuracy
        best_epoch = epoch

    logger.append([epoch, val_accuracy, test_accuracy])
    print('epoch%d  val accu, test accu:'%(epoch), val_accuracy, test_accuracy)
    
logger.close()

print("Ablation study %d examples"%select_num, 'best_val epoch: %d'%best_epoch, \
      test_best, 'last', test_accuracy, 'potential_best_test', np.max(test_accu))

    
###################### Step 4: Using Weighted MixMatch #######################

# hyper-parameter record: sel_epoch, beta, gamma_index, kappa
with open("./hyperparameter_record.txt", 'a') as f:
    f.writelines("\n")
    import datetime
    now_time = datetime.datetime.now()
    line = datetime.datetime.strftime(now_time,'%Y-%m-%d %H:%M:%S')+"\n"
    f.writelines(line)
    line = args.dataset + " noise rate: " + args.noise_rate + " " + args.sel_epoch + "\n"
    f.writelines(line)
    line = "beta: %.2f "%args.beta + "gamma: %.2f "%args.gamma +  "kappa: %.2f\n"%args.kappa
    f.writelines(line)
    f.writelines(line)

    
from MixMatch_train_CIFAR import Weighted_MixMatch_main
folder = 'selected_noisy_data/%s_%.1f'%(args.dataset, args.noise_rate)


labeled_file = np.load(folder+"/%d_labeled_ind_%.1f_sel_epoch_%d.npy"%(select_num, args.noise_rate, args.sel_epoch))
weight_list = np.load(folder+"/%d_w_list_%.1f_sel_epoch_%d.npy"%(select_num, args.noise_rate, args.sel_epoch)) # Weighting sheme
unlabeled_file = np.load(folder+"/%d_unlabeled_ind_%.1f_sel_epoch_%d.npy"%(select_num, args.noise_rate, args.sel_epoch))
val_file = np.load(folder+"/%d_val_ind_%.1f_sel_epoch_%d.npy"%(select_num, args.noise_rate, args.sel_epoch))
noisy_label50000_file = np.load(folder+"/%d_noisy_label_50000_%.1f_sel_epoch_%d.npy"%(select_num, args.noise_rate, args.sel_epoch))

all_data_information = [labeled_file, weight_list, unlabeled_file, val_file, noisy_label50000_file]

Weighted_MixMatch_main(all_data_information, args.dataset, args.noise_rate, select_num)

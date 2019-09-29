# -*- coding: utf-8 -*-
import os
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

from models.classification_heads import ClassificationHead
from models.R2D2_embedding import R2D2Embedding
from models.protonet_embedding import ProtoNetEmbedding
from models.ResNet12_embedding import resnet12

from utils import set_gpu, Timer, count_accuracy, check_dir, log, AttackPGDFeatureExtractor

def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.
        
    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1,index,1)
    
    return encoded_indicies

def get_model(options):
    # Choose the embedding network & corresponding linear head
    if options.network == 'ProtoNet':
        network = ProtoNetEmbedding().cuda()
        cls_head = torch.nn.Linear(1600, 64).cuda()
    elif options.network == 'R2D2':
        network = R2D2Embedding().cuda()
        cls_head = torch.nn.Linear(51200, 64).cuda()
    elif options.network == 'ResNet':
        if options.dataset == 'miniImageNet' or options.dataset == 'tieredImageNet':
            network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=5).cuda()
            network = torch.nn.DataParallel(network, device_ids=list(map(int, (options.gpu).split(" ,"))))
        else:
            network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=2).cuda()
        cls_head = torch.nn.Linear(16000, 64).cuda()
    else:
        print ("Cannot recognize the network type")
        assert(False)
        
    return (network, cls_head)

def get_dataset(options):
    # Choose the embedding network
    if options.dataset == 'miniImageNet':
        from data.mini_imagenet import MiniImageNet
        from torch.utils.data import DataLoader
        dataset_train = MiniImageNet(phase='train')
        dataset_val = MiniImageNet(phase='val')
        data_loader = DataLoader
    elif options.dataset == 'tieredImageNet':
        from data.tiered_imagenet import tieredImageNet
        from torch.utils.data import DataLoader
        dataset_train = tieredImageNet(phase='train')
        dataset_val = tieredImageNet(phase='val')
        data_loader = DataLoader
    elif options.dataset == 'CIFAR_FS':
        from data.CIFAR_FS import CIFAR_FS
        from torch.utils.data import DataLoader
        dataset_train = CIFAR_FS(phase='train')
        dataset_val = CIFAR_FS(phase='val')
        data_loader = DataLoader
    elif options.dataset == 'FC100':
        from data.FC100 import FC100
        from torch.utils.data import DataLoader
        dataset_train = FC100(phase='train')
        dataset_val = FC100(phase='val')
        data_loader = DataLoader
    else:
        print ("Cannot recognize the dataset type")
        assert(False)
        
    return (dataset_train, dataset_val, data_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epoch', type=int, default=60,
                            help='number of training epochs')
    parser.add_argument('--save-epoch', type=int, default=10,
                            help='frequency of model saving')
    parser.add_argument('--save-path', default='./experiments/exp_1')
    parser.add_argument('--train_batch_size', type=int, default=64,
                            help='training batch size')
    parser.add_argument('--val_batch_size', type=int, default=64,
                            help='val batch size')
    parser.add_argument('--ways', type=int, default=64,
                            help='number of classes')
    parser.add_argument('--gpu', default='0, 1, 2, 3')
    parser.add_argument('--network', type=str, default='ProtoNet',
                            help='choose which embedding network to use. ProtoNet, R2D2, ResNet')
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                            help='choose which classification head to use. miniImageNet, tieredImageNet, CIFAR_FS, FC100')
    parser.add_argument('--eps', type=float, default=0.0,
                            help='epsilon of label smoothing')
    parser.add_argument('--lr', type=float, default=0.001,
                            help='initial learning rate')
    parser.add_argument('--attack_embedding', action='store_true',
                            help='use attacks to train embedding?')
    parser.add_argument('--attack_epsilon', type=float, default=8.0/255.0,
                            help='epsilon for linfinity ball in which images are perturbed')
    parser.add_argument('--attack_steps', type=int, default=3,
                            help='number of PGD steps for each attack')
    parser.add_argument('--attack_step_size', type=float, default=2.0/255.0,
                            help='number of query examples per training class')
    parser.add_argument('--attack_targeted', action='store_true',
                            help='used targeted attacks')

    opt = parser.parse_args()

    (dataset_train, dataset_val, data_loader) = get_dataset(opt)

    config = {
    'epsilon': opt.attack_epsilon,
    'num_steps': opt.attack_steps,
    'step_size': opt.attack_step_size,
    'targeted': opt.attack_targeted,
    'random_init': True
    }

    # Dataloader of Gidaris & Komodakis (CVPR 2018)
    dloader_train = data_loader(
        dataset=dataset_train,
        batch_size=opt.train_batch_size,
        num_workers=4,
        shuffle=True
    )

    dloader_val = data_loader(
        dataset=dataset_val,
        batch_size=opt.val_batch_size,
        num_workers=4
    )

    set_gpu(opt.gpu)
    check_dir('./experiments/')
    check_dir(opt.save_path)
    
    log_file_path = os.path.join(opt.save_path, "train_log.txt")
    log(log_file_path, str(vars(opt)))

    (embedding_net, cls_head) = get_model(opt)
    
    optimizer = torch.optim.SGD([{'params': embedding_net.parameters()}, 
                                 {'params': cls_head.parameters()}], lr=opt.lr, momentum=0.9, \
                                          weight_decay=5e-4, nesterov=True)
    
    lambda_epoch = lambda e: 1.0 if e < 20 else (0.06 if e < 40 else 0.012 if e < 50 else (0.0024))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch, last_epoch=-1)

    max_train_acc = 0.0

    timer = Timer()
    x_entropy = torch.nn.CrossEntropyLoss()
    
    for epoch in range(1, opt.num_epoch + 1):
        # Train on the training split
        lr_scheduler.step()
        
        # Fetch the current epoch's learning rate
        epoch_learning_rate = 0.1
        for param_group in optimizer.param_groups:
            epoch_learning_rate = param_group['lr']
            
        log(log_file_path, 'Train Epoch: {}\tLearning Rate: {:.4f}'.format(
                            epoch, epoch_learning_rate))
        
        _, _ = [x.train() for x in (embedding_net, cls_head)]
        
        train_accuracies = []
        train_losses = []

        for i, (X, y) in enumerate(tqdm(dloader_train)):
            X = X.cuda()
            y = y.cuda()

            
            X_adv = AttackPGDFeatureExtractor(opt.attack_embedding, embedding_net, cls_head, config, X, y)
            emb_query = embedding_net(X_adv)
            logit_query = cls_head(emb_query)

            smoothed_one_hot = one_hot(y, opt.ways)
            smoothed_one_hot = smoothed_one_hot * (1 - opt.eps) + (1 - smoothed_one_hot) * opt.eps / (opt.ways - 1)

            log_prb = F.log_softmax(logit_query, dim=1)
            loss = -(smoothed_one_hot * log_prb).sum(dim=1)
            loss = loss.mean()
            
            acc = count_accuracy(logit_query, y)
            
            train_accuracies.append(acc.item())
            train_losses.append(loss.item())

            if (i % 100 == 0):
                train_acc_avg = np.mean(np.array(train_accuracies))
                log(log_file_path, 'Train Epoch: {}\tBatch: [{}/{}]\tLoss: {:.4f}\tAccuracy: {:.2f} % ({:.2f} %)'.format(
                            epoch, i, len(dloader_train), loss.item(), train_acc_avg, acc))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_acc_avg = np.mean(np.array(train_accuracies))
        train_loss_avg = np.mean(np.array(train_losses))

        if train_acc_avg > max_train_acc:
            max_train_acc = train_acc_avg
            torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()},\
                       os.path.join(opt.save_path, 'best_model.pth'))
            log(log_file_path, 'Train Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} % (Best)'\
                  .format(epoch, train_loss_avg, train_acc_avg))
        else:
            log(log_file_path, 'Train Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} %'\
                  .format(epoch, train_loss_avg, train_acc_avg))

        torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()}\
                   , os.path.join(opt.save_path, 'last_epoch.pth'))

        if epoch % opt.save_epoch == 0:
            torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()}\
                       , os.path.join(opt.save_path, 'epoch_{}.pth'.format(epoch)))

        log(log_file_path, 'Elapsed Time: {}/{}\n'.format(timer.measure(), timer.measure(epoch / float(opt.num_epoch))))

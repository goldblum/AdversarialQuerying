import os
import time
import pprint
import torch
import torch.nn.functional as F
import random

def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)

def check_dir(path):
    '''
    Create directory if it does not exist.
        path:           Path of directory.
    '''
    if not os.path.exists(path):
        os.mkdir(path)

def count_accuracy(logits, label):
    pred = torch.argmax(logits, dim=1).view(-1)
    label = label.view(-1)
    accuracy = 100 * pred.eq(label).float().mean()
    return accuracy

class Timer():
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / float(p)
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

def log(log_file_path, string):
    '''
    Write one line of log into screen and file.
        log_file_path: Path of log file.
        string:        String to write in log file.
    '''
    with open(log_file_path, 'a+') as f:
        f.write(string + '\n')
        f.flush()
    print(string)

def AttackPGD(attack, embedding_net, cls_head, config, data_query, emb_support, labels_query, labels_support, way, shot, head, episodes_per_batch, n_query, maxIter = 3):
    if not attack:
        return data_query
    if config['targeted']:
        new_labels_query = torch.zeros_like(labels_query)
        for i in range(int(labels_query.size()[0])):
            for j in range(int(labels_query.size()[1])):
                while True:
                    new_labels_query[i,j] = random.randint(0,way-1)
                    if new_labels_query[i,j] != labels_query[i,j]:
                        break
    else:
        new_labels_query = labels_query
    new_labels_query = new_labels_query.view(new_labels_query.size()[0]*new_labels_query.size()[1])
    x = data_query.detach()
    if config['random_init']:
        x = x + torch.zeros_like(x).uniform_(-config['epsilon'], config['epsilon'])
    for i in range(config['num_steps']):
        x.requires_grad_()
        with torch.enable_grad():
            emb_query_adv = embedding_net(x.reshape([-1] + list(x.shape[-3:]))).reshape(episodes_per_batch, n_query, -1)

            if head == 'SVM':
                logits = cls_head(emb_query_adv, emb_support, labels_support, way, shot, maxIter=maxIter)
            else:
                logits = cls_head(emb_query_adv, emb_support, labels_support, way, shot)

            logits = logits.view(logits.size()[0]*logits.size()[1],logits.size()[2])
            loss = F.cross_entropy(logits, new_labels_query, size_average=False)
        grad = torch.autograd.grad(loss, [x])[0]
        if config['targeted']:
            x = x.detach() - config['step_size']*torch.sign(grad.detach())
        else:
            x = x.detach() + config['step_size']*torch.sign(grad.detach())
        x = torch.min(torch.max(x, data_query - config['epsilon']), data_query + config['epsilon'])
        x = torch.clamp(x, 0.0, 1.0)
    return x

def AttackPGDFeatureExtractor(attack, embedding_net, cls_head, config, data_query, labels_query, ways=64, maxIter = 3):
    if not attack:
        return data_query
    if config['targeted']:
        new_labels_query = torch.zeros_like(labels_query)
        for i in range(int(labels_query.size()[0])):
            while True:
                new_labels_query[i] = random.randint(0,ways-1)
                if new_labels_query[i] != labels_query[i]:
                    break
    else:
        new_labels_query = labels_query

    x = data_query.detach()
    if config['random_init']:
        x = x + torch.zeros_like(x).uniform_(-config['epsilon'], config['epsilon'])
    for i in range(config['num_steps']):
        x.requires_grad_()
        with torch.enable_grad():
            emb_query_adv = embedding_net(X)
            logits = cls_head(emb_query_adv)
            loss = F.cross_entropy(logits, new_labels_query)
        grad = torch.autograd.grad(loss, [x])[0]
        if config['targeted']:
            x = x.detach() - config['step_size']*torch.sign(grad.detach())
        else:
            x = x.detach() + config['step_size']*torch.sign(grad.detach())
        x = torch.min(torch.max(x, data_query - config['epsilon']), data_query + config['epsilon'])
        x = torch.clamp(x, 0.0, 1.0)
    return x

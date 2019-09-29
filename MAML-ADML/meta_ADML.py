import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np

from    learner import Learner
from    copy import deepcopy

def zero_nontrainable_grads(grads, trainable_layers=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]):
    for index, grad_tensor in enumerate(grads):
        if index not in trainable_layers:
            grad_tensor = torch.zeros_like(grad_tensor)

def inputsPGD(metalearner, net, inputs, targets, params = False, evaluate = False):
    if evaluate:
        attack_steps = metalearner.eval_attack_steps
        attack_step_size = metalearner.eval_attack_step_size
    else:
        attack_steps = metalearner.attack_steps
        attack_step_size = metalearner.attack_step_size
    x = inputs.detach()
    if not metalearner.no_random_start:
        x = x + torch.zeros_like(x).uniform_(-metalearner.attack_epsilon, metalearner.attack_epsilon)
    for i in range(attack_steps):
        x.requires_grad_()
        with torch.enable_grad():
            if params:
                loss = F.cross_entropy(net(x, params), targets, size_average=False)
            else:
                loss = F.cross_entropy(net(x), targets, size_average=False)
        grad = torch.autograd.grad(loss, [x])[0]
        if metalearner.targeted:
            x = x.detach() - attack_step_size*torch.sign(grad.detach())
        else:
            x = x.detach() + attack_step_size*torch.sign(grad.detach())
        x = torch.min(torch.max(x, inputs - metalearner.attack_epsilon), inputs + metalearner.attack_epsilon)
        x = torch.clamp(x, 0.0, 1.0)
    return x

class Meta_ADML(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config):
        """

        :param args:
        """
        super(Meta_ADML, self).__init__()
        
        self.finetune_trainable = args.finetune_trainable
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.attack_query = args.attack_query
        self.attack_support = args.attack_support
        self.no_attack_validation = args.no_attack_validation
        self.attack_epsilon = args.attack_epsilon
        self.attack_step_size = args.attack_step_size
        self.attack_steps = args.attack_steps
        self.eval_attack_steps = args.eval_attack_steps
        self.eval_attack_step_size = args.eval_attack_step_size
        self.net = Learner(config, args.imgc, args.imgsz)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        self.no_random_start = args.no_random_start
        self.targeted = args.targeted

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter


    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """

        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)
        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]    

        for i in range(task_num):

            # 1. run the i-th task and compute loss for k=0
            
            logits_robust = self.net(inputsPGD(self, self.net, x_spt[i], y_spt[i]), vars=None, bn_training=True)
            logits_natural = self.net(x_spt[i], vars=None, bn_training=True)
            loss_natural = F.cross_entropy(logits_natural, y_spt[i])
            loss_robust = F.cross_entropy(logits_robust, y_spt[i])
            grad_natural = torch.autograd.grad(loss_natural, self.net.parameters())
            zero_nontrainable_grads(grad_natural, trainable_layers=self.finetune_trainable)
            grad_robust = torch.autograd.grad(loss_robust, self.net.parameters())
            zero_nontrainable_grads(grad_robust, trainable_layers=self.finetune_trainable)
            fast_weights_natural = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_natural, self.net.parameters())))
            fast_weights_robust = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_robust, self.net.parameters())))

            # this is the loss and accuracy before first update

            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update

            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], fast_weights_robust, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[1] += loss_q
                # [setsz]
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits_robust = self.net(inputsPGD(self, self.net, x_spt[i], y_spt[i], params = fast_weights_robust), fast_weights_robust, bn_training=True)
                logits_natural = self.net(x_spt[i], fast_weights_natural, bn_training=True)
                loss_natural = F.cross_entropy(logits_natural, y_spt[i])
                loss_robust = F.cross_entropy(logits_robust, y_spt[i])
                
                # 2. compute grad on theta_pi
                grad_natural = torch.autograd.grad(loss_natural, fast_weights_natural)
                zero_nontrainable_grads(grad_natural, trainable_layers=self.finetune_trainable)
                grad_robust = torch.autograd.grad(loss_robust, fast_weights_robust)
                zero_nontrainable_grads(grad_robust, trainable_layers=self.finetune_trainable)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights_natural = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_natural, fast_weights_natural)))
                fast_weights_robust = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_robust, fast_weights_robust)))

                logits_q_robust = self.net(inputsPGD(self, self.net, x_qry[i], y_qry[i], params = fast_weights_natural), fast_weights_natural, bn_training=True)
                logits_q_natural = self.net(x_qry[i], fast_weights_robust, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q_natural = F.cross_entropy(logits_q_natural, y_qry[i])
                loss_q_robust = F.cross_entropy(logits_q_robust, y_qry[i])
                losses_q[k + 1] += loss_q_natural+loss_q_robust

                with torch.no_grad():
                    pred_q = F.softmax(logits_q_robust, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct



        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = 0.5*losses_q[-1] / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # 	print(torch.norm(p).item())
        self.meta_optim.step()


        accs = np.array(corrects) / (querysz * task_num)

        return accs

    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4
        print('Validating...')

        querysz = x_qry.size(0)

        natural_corrects = [0 for _ in range(self.update_step_test + 1)]
        robust_corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        zero_nontrainable_grads(grad, trainable_layers=self.finetune_trainable)
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            natural_correct = torch.eq(pred_q, y_qry).sum().item()
            natural_corrects[0] = natural_corrects[0] + natural_correct

            # [setsz, nway]
            robust_logits_q = net(inputsPGD(self, net, x_qry, y_qry, net.parameters(), evaluate=True), net.parameters(), bn_training=True)
            # [setsz]
            robust_pred_q = F.softmax(robust_logits_q, dim=1).argmax(dim=1)
            # scalar
            robust_correct = torch.eq(robust_pred_q, y_qry).sum().item()
            robust_corrects[0] = robust_corrects[0] + robust_correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            natural_corrects[1] = natural_corrects[1] + natural_correct

            # [setsz, nway]
            robust_logits_q = net(inputsPGD(self, net, x_qry, y_qry, fast_weights, evaluate=True), fast_weights, bn_training=True)
            # [setsz]
            robust_pred_q = F.softmax(robust_logits_q, dim=1).argmax(dim=1)
            # scalar
            robust_correct = torch.eq(robust_pred_q, y_qry).sum().item()
            robust_corrects[1] = robust_corrects[1] + robust_correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            zero_nontrainable_grads(grad, trainable_layers=self.finetune_trainable)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                natural_correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                natural_corrects[k + 1] = natural_corrects[k + 1] + natural_correct

            robust_logits_q = net(inputsPGD(self, net, x_qry, y_qry, fast_weights, evaluate=True), fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            robust_loss_q = F.cross_entropy(robust_logits_q, y_qry)

            with torch.no_grad():
                robust_pred_q = F.softmax(robust_logits_q, dim=1).argmax(dim=1)
                robust_correct = torch.eq(robust_pred_q, y_qry).sum().item()  # convert to numpy
                robust_corrects[k + 1] = robust_corrects[k + 1] + robust_correct

        del net

        natural_accs = np.array(natural_corrects) / querysz
        robust_accs = np.array(robust_corrects) / querysz


        ########################### DO THE SAME THING BUT ADVERSARIALLY TRAINED ON SUPPORT ########################

        natural_corrects = [0 for _ in range(self.update_step_test + 1)]
        robust_corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(inputsPGD(self, net, x_spt, y_spt), bn_training=True)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        zero_nontrainable_grads(grad, trainable_layers=self.finetune_trainable)
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            natural_correct = torch.eq(pred_q, y_qry).sum().item()
            natural_corrects[0] = natural_corrects[0] + natural_correct

            # [setsz, nway]
            robust_logits_q = net(inputsPGD(self, net, x_qry, y_qry, net.parameters(), evaluate=True), net.parameters(), bn_training=True)
            # [setsz]
            robust_pred_q = F.softmax(robust_logits_q, dim=1).argmax(dim=1)
            # scalar
            robust_correct = torch.eq(robust_pred_q, y_qry).sum().item()
            robust_corrects[0] = robust_corrects[0] + robust_correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            natural_corrects[1] = natural_corrects[1] + natural_correct

            # [setsz, nway]
            robust_logits_q = net(inputsPGD(self, net, x_qry, y_qry, fast_weights, evaluate=True), fast_weights, bn_training=True)
            # [setsz]
            robust_pred_q = F.softmax(robust_logits_q, dim=1).argmax(dim=1)
            # scalar
            robust_correct = torch.eq(robust_pred_q, y_qry).sum().item()
            robust_corrects[1] = robust_corrects[1] + robust_correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(inputsPGD(self, net, x_spt, y_spt, params = fast_weights), fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            zero_nontrainable_grads(grad, trainable_layers=self.finetune_trainable)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                natural_correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                natural_corrects[k + 1] = natural_corrects[k + 1] + natural_correct

            robust_logits_q = net(inputsPGD(self, net, x_qry, y_qry, fast_weights, evaluate=True), fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            robust_loss_q = F.cross_entropy(robust_logits_q, y_qry)

            with torch.no_grad():
                robust_pred_q = F.softmax(robust_logits_q, dim=1).argmax(dim=1)
                robust_correct = torch.eq(robust_pred_q, y_qry).sum().item()  # convert to numpy
                robust_corrects[k + 1] = robust_corrects[k + 1] + robust_correct

        del net

        natural_accs_advTrained = np.array(natural_corrects) / querysz
        robust_accs_advTrained = np.array(robust_corrects) / querysz

        return natural_accs, robust_accs, natural_accs_advTrained, robust_accs_advTrained





def main():
    pass


if __name__ == '__main__':
    main()

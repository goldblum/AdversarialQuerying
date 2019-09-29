import  torch, os
import  numpy as np
from    MiniImagenet import MiniImagenet
import  scipy.stats
from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
import  random, sys, pickle
import  argparse

from meta import Meta
from meta_ADML import Meta_ADML


def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h

def main():

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    #print(args)

    config = [
        ('conv2d', [32, 3, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 1, 0]),
        ('flatten', []),
        ('linear', [args.n_way, 32 * 5 * 5])
    ]

    device = torch.device('cuda')
    #device = torch.device('cpu')

    if args.ADML:
        maml = Meta_ADML(args,config).to(device)
    else:
        maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    #print(maml)
    #print('Total trainable tensors:', num)

    # batchsz here means total episode number
    mini = MiniImagenet('../MAML-Mini-ImageNet/', mode='train', n_way=args.n_way, k_shot=args.k_spt,
                        k_query=args.k_qry,
                        batchsz=10000, resize=args.imgsz)
    mini_test = MiniImagenet('../MAML-Mini-ImageNet/', mode='test', n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry,
                             batchsz=100, resize=args.imgsz)

    if not os.path.isdir(args.save_path+'/mini-imagenet/'):
        os.makedirs(args.save_path+'/mini-imagenet/', )

    natural_validation_accuracy = 0.0
    robust_validation_accuracy = 0.0
    natural_validation_accuracy_advTrained = 0.0
    robust_validation_accuracy_advTrained = 0.0
    for epoch in range(args.epoch//10000):
        # fetch meta_batchsz num of episode each time
        db = DataLoader(mini, args.task_num, shuffle=True, num_workers=1, pin_memory=True)

        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
            accs = maml(x_spt, y_spt, x_qry, y_qry)

            if (step+1) % 30 == 0:
                print('step:', step, 'training acc:', accs)

            if (step+1) % len(db) == 0:  # evaluation
                db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=1, pin_memory=True)
                accs_all_test = []
                robust_accs_all_test = []
                accs_all_test_advTrained = []
                robust_accs_all_test_advTrained = []
                for x_spt, y_spt, x_qry, y_qry in db_test:
                    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                 x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

                    accs, robust_accs, accs_advTrained, robust_accs_advTrained = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
                    accs_all_test.append(accs)
                    robust_accs_all_test.append(robust_accs)
                    accs_all_test_advTrained.append(accs_advTrained)
                    robust_accs_all_test_advTrained.append(robust_accs_advTrained)

                # [b, update_step+1]
                accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                robust_accs = np.array(robust_accs_all_test).mean(axis=0).astype(np.float16)
                accs_advTrained = np.array(accs_all_test_advTrained).mean(axis=0).astype(np.float16)
                robust_accs_advTrained = np.array(robust_accs_all_test_advTrained).mean(axis=0).astype(np.float16)
                natural_validation_accuracy = 100.0*accs[-1]
                robust_validation_accuracy = 100.0*robust_accs[-1]
                natural_validation_accuracy_advTrained = 100.0*accs_advTrained[-1]
                robust_validation_accuracy_advTrained = 100.0*robust_accs_advTrained[-1]
                print('Test natural acc:', accs)
                print('Test robust acc:', robust_accs)
                print('Test natural acc when adversarially fine tuned: ', accs_advTrained)
                print('Test robust acc when adversarially fine tuned:', robust_accs_advTrained)


        print('\nSaving..')
        state = {
            'net_params': maml.net.parameters()
        }
        torch.save(state, args.save_path+'/mini-imagenet/'+'/'+str(args.k_spt)+'_shot_epoch='+str(epoch)+'.t7')
    f = open(args.save_path +'/mini-imagenet/'+str(args.k_spt)+'_shot_test_acc.txt', 'w+')
    f.write('natural acc: '+str(natural_validation_accuracy)+', robust accuracy: '+str(robust_validation_accuracy)+', natural accuracy adversarially fine tuned: '+str(natural_validation_accuracy_advTrained)+', robust accuracy adversarially fine tuned: '+str(robust_validation_accuracy_advTrained))
    f.close()    

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetuning', default=10)
    argparser.add_argument('--attack_query', action='store_true', help='attack query')
    argparser.add_argument('--attack_support', action='store_true', help='attack support')
    argparser.add_argument('--attack_epsilon', type=float, help='maximum attack norm', default=8.0/255.0)
    argparser.add_argument('--no_attack_validation', action='store_true', help='no attack during validation')
    argparser.add_argument('--attack_step_size', type=float, help='step size for attacker', default=8.0/255.0)
    argparser.add_argument('--attack_steps', type=int, help='number of attack steps', default=1)
    argparser.add_argument('--eval_attack_steps', type=int, help='number of attack steps', default=20)
    argparser.add_argument('--eval_attack_step_size', type=float, help='number of attack steps', default=2.0/255)
    argparser.add_argument('--no_random_start', action='store_true', help='number of validation attack steps')
    argparser.add_argument('--targeted', action='store_true', help='targeted attacks')
    argparser.add_argument('--save_path', default='checkpoint/', help='path to save models and stats')
    argparser.add_argument('--ADML', action='store_true', help='use adversarial meta-learning')
    argparser.add_argument('--finetune_trainable', nargs='+', default=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17], type=int, help='which blocks to train during finetuning')

    args = argparser.parse_args()

    main()

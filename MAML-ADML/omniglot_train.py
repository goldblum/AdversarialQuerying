import  torch, os
import  numpy as np
from    omniglotNShot import OmniglotNShot
import  argparse

from    meta import Meta
from meta_ADML import Meta_ADML

def main(args):

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    #print(args)

    config = [
        ('conv2d', [64, 1, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 2, 2, 1, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('flatten', []),
        ('linear', [args.n_way, 64])
    ]

    device = torch.device('cuda')
    if args.ADML:
        maml = Meta_ADML(args,config).to(device)
    else:
        maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    #print(maml)
    #print('Total trainable tensors:', num)

    db_train = OmniglotNShot('omniglot',
                       batchsz=args.task_num,
                       n_way=args.n_way,
                       k_shot=args.k_spt,
                       k_query=args.k_qry,
                       imgsz=args.imgsz)

    if not os.path.isdir(args.save_path+'/omniglot/'):
        os.makedirs(args.save_path+'/omniglot/', )

    natural_validation_accuracy = 0.0
    robust_validation_accuracy = 0.0
    natural_validation_accuracy_advTrained = 0.0
    robust_validation_accuracy_advTrained = 0.0

    for step in range(args.epoch):

        x_spt, y_spt, x_qry, y_qry = db_train.next()
        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                     torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

        # set traning=True to update running_mean, running_variance, bn_weights, bn_bias
        accs = maml(x_spt, y_spt, x_qry, y_qry)

        if (step+1) % 50 == 0:
            print('step:', step, '\ttraining acc:', accs)

        if (step+1) % 1000 == 0:
            accs = []
            robust_accs = []
            accs_advTrained = []
            robust_accs_advTrained = []
            for _ in range(1000//args.task_num):
                # test
                x_spt, y_spt, x_qry, y_qry = db_train.next('test')
                x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                             torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

                # split to single task each time
                for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                    test_acc, robust_test_acc, test_acc_advTrained, robust_test_acc_advTrained = maml.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                    accs.append( test_acc )
                    robust_accs.append( robust_test_acc )
                    accs_advTrained.append( test_acc_advTrained )
                    robust_accs_advTrained.append( robust_test_acc_advTrained )

            # [b, update_step+1]
            accs = np.array(accs).mean(axis=0).astype(np.float16)
            robust_accs = np.array(robust_accs).mean(axis=0).astype(np.float16)
            accs_advTrained = np.array(accs_advTrained).mean(axis=0).astype(np.float16)
            robust_accs_advTrained = np.array(robust_accs_advTrained).mean(axis=0).astype(np.float16)
            natural_validation_accuracy = 100.0*accs[-1]
            robust_validation_accuracy = 100.0*robust_accs[-1]
            natural_validation_accuracy_advTrained = 100.0*accs_advTrained[-1]
            robust_validation_accuracy_advTrained = 100.0*robust_accs_advTrained[-1]
            print('Test Natural acc:', accs)
            print('Test Robust acc:', robust_accs)
            print('Test Natural acc adversarially fine tuned:', accs_advTrained)
            print('Test Robust acc adversarially fine tuned:', robust_accs_advTrained)

            print('\nSaving..')
            state = {
                'net_params': maml.net.parameters()
            }
            torch.save(state, args.save_path+'/omniglot/'+str(args.k_spt)+'_shot_epoch='+str(step)+'.t7')

    f = open(args.save_path +'/omniglot/'+str(args.k_spt)+'_shot_test_acc.txt', 'w+')
    f.write('natural acc: '+str(natural_validation_accuracy)+', robust accuracy: '+str(robust_validation_accuracy)+', natural acc when adversarially fine tuned: '+str(natural_validation_accuracy_advTrained)+', robust accuracy when adversarially fine tuned: '+str(robust_validation_accuracy_advTrained))
    f.close()


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=4000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    argparser.add_argument('--imgc', type=int, help='imgc', default=1)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.4)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--attack_query', action='store_true', help='attack query')
    argparser.add_argument('--attack_support', action='store_true', help='attack support')
    argparser.add_argument('--attack_epsilon', type=float, help='maximum attack norm', default=16.0/255.0)
    argparser.add_argument('--no_attack_validation', action='store_true', help='no attack during validation')
    argparser.add_argument('--attack_step_size', type=float, help='step size for attacker', default=16.0/255.0)
    argparser.add_argument('--attack_steps', type=int, help='number of attack steps', default=1)
    argparser.add_argument('--eval_attack_steps', type=int, help='number of validation attack steps', default=20)
    argparser.add_argument('--eval_attack_step_size', type=float, help='number of validation attack steps', default=4.0/255)
    argparser.add_argument('--no_random_start', action='store_true', help='number of attack steps')
    argparser.add_argument('--targeted', action='store_true', help='targeted attacks')
    argparser.add_argument('--save_path', default='checkpoint/', help='path to save models and stats')
    argparser.add_argument('--ADML', action='store_true', help='use adversarial meta-learning')


    args = argparser.parse_args()

    main(args)

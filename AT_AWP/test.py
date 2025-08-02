"""Training file for the victim models"""
import os
import argparse
import sys
sys.path.append('../')
sys.path.append('./')
import torch
import pickle
import torch.nn as nn
from baselines.config import BEST_WEIGHTS
from baselines.config import MAX_PERTURB_BATCH as BATCH_SIZE
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from baselines.attack import ChamferDist,ChamferkNNDhusdorfist, HausdorffDist, ChamferL2
from baselines.dataset import ModelNet40, ModelNetDataLoader
from baselines.model import DGCNN, PointNetCls, \
    PointNet2ClsSsg, PointConvDensityClsSsg, Pct
from torch.utils.data import DataLoader
from baselines.util.utils import str2bool, set_seed
from baselines.attack.CW import UAE_pretrain
import numpy as np
from baselines.attack import ClipLinf
from tqdm import tqdm
from baselines.latent_3d_points.AE_z import AutoEncoder_z
from baselines.model import DGCNN, PointNetCls, PointNet2ClsSsg, PointConvDensityClsSsg, Pct, PointTransformerCls
from baselines.util.utils import str2bool, set_seed
from baselines.attack import CWPerturb
from baselines.attack import CrossEntropyAdvLoss, UntargetedLogitsAdvLoss
from baselines.attack import L2Dist
from baselines.attack import ClipPointsLinf
from baselines.attack  import CWUKNN
from baselines.attack  import ChamferkNNDist
from baselines.attack  import ProjectInnerClipLinf
from baselines.latent_3d_points.src import encoders_decoders
from baselines.attack import CWUAdvPC
from baselines.attack import CW_PFattack
from baselines.attack import AdvCrossEntropyLoss
from baselines.attack import ChamferDist, HausdorffDist
import matplotlib
from Landscape4Input import Landscape4Input
from torch.nn import functional as F
from matplotlib import pyplot as plt
from tqdm.contrib.itertools import product
matplotlib.use('Agg')


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--data_root', type=str,
                        default='baselines/official_data/modelnet40_normal_resampled')
    parser.add_argument('--feature_transform', type=str2bool, default=True,
                        help='whether to use STN on features in PointNet')
    parser.add_argument('--dataset', type=str, default='mn40',
                        choices=['mn40'])
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings in DGCNN')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use in DGCNN')
    parser.add_argument('--print_iter', type=int, default=50,
                        help='Print interval')
    parser.add_argument('--num_point', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    
    ## pct
    parser.add_argument('--dropout', type=float, default=0.5,help='dropout rate')

    ## 3D-Adv
    parser.add_argument('--adv_func', type=str, default='logits',
                        choices=['logits', 'cross_entropy'],
                        help='Adversarial loss function to use')
    parser.add_argument('--kappa_3d_adv', type=float, default=30.,
                        help='min margin in logits adv loss')
    parser.add_argument('--attack_lr_3d_adv', type=float, default=1e-2,
                        help='lr in CW optimization')
    parser.add_argument('--binary_step_3d_adv', type=int, default=10, metavar='N',
                        help='Binary search step') #10
    parser.add_argument('--num_iter_3d_adv', type=int, default=500, metavar='N',
                        help='Number of iterations in each search step') #500
    
    #### knn 
    parser.add_argument('--kappa_knn', type=float, default=15.,
                        help='min margin in logits adv loss')
    parser.add_argument('--attack_lr_knn', type=float, default=1e-3,
                        help='lr in CW optimization')
    parser.add_argument('--num_iter_knn', type=int, default=2500, metavar='N',
                        help='Number of iterations in each search step')
    
    ### advpc
    parser.add_argument('--ae_model_path', type=str,
                        default='baselines/latent_3d_points/src/logs/mn40/AE/BEST_model9800_CD_0.0038.pth')
    parser.add_argument('--kappa_advpc', type=float, default=30.,
                        help='min margin in logits adv loss')
    parser.add_argument('--GAMMA_advpc', type=float, default=0.25,
                        help='hyperparameter gamma')
    parser.add_argument('--binary_step_advpc', type=int, default=2, metavar='N',
                        help='Number of binary search step')
    parser.add_argument('--attack_lr_advpc', type=float, default=1e-2,
                        help='lr in attack training optimization')
    parser.add_argument('--num_iter_advpc', type=int, default=200, metavar='N',
                        help='Number of iterations in each search step')
    
    ### pf-attack
    parser.add_argument('--adv_func_pf', type=str, default='cross_entropy',
                        choices=['logits', 'cross_entropy'],
                        help='Adversarial loss function to use')
    parser.add_argument('--drop_rate', type=float, default=0.01,
                        help='drop rate of points')
    parser.add_argument('--t_pf', type=float, default=1.0,
                        choices=[1.0, 0.2])
    parser.add_argument('--players', type=int, default=64, metavar='N',
                        help='num of players')
    parser.add_argument('--k_sharp', type=int, default=54, metavar='N',
                        help='num of k_sharp')
    parser.add_argument('--attack_lr_pf', type=float, default=1e-2,
                        help='lr in CW optimization')
    parser.add_argument('--binary_step_pf', type=int, default=1, metavar='N',
                        help='Binary search step')
    parser.add_argument('--num_iter_pf', type=int, default=200, metavar='N',
                        help='Number of iterations in each search step')
    parser.add_argument('--initial_const_pf', type=float, default=10, help='')
    parser.add_argument('--pp_pf', type=float, default=0.5, help='生成随机数为1的概率')
    parser.add_argument('--trans_weight_pf', type=float, default=0.5, help='')


    ####### parameters of generator ########
    parser.add_argument('--finetune_mode', type=str, default='noise',
                        choices=['noise', 'G'])
    parser.add_argument('--attack_methods', type=str, default='PF-attack',
                        choices=['3D-Adv', 'KNN', 'AdvPC', 'PF-attack'])
    parser.add_argument('--attack_type', type=str, default='both',
                        choices=['both', 'single'])
    parser.add_argument('--init_noise', default=True, help='use normals')
    parser.add_argument('--batch_size', type=int, default=3,
                        help='Size of batch')
    parser.add_argument('--test_batch_size', type=int, default=1,
                        help='Size of batch')
    parser.add_argument('--epoch', type=int, default=200, metavar='N',
                        help='number of epochs to train ')
    parser.add_argument('--epoch_test', type=int, default=50, metavar='N',
                        help='number of epochs to train ')
    parser.add_argument('--binary_step', type=int, default=4, metavar='N',
                        help='number of epochs to train ')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate for the optimizer')
    parser.add_argument('--model', type=str, default='pointnet',
                        choices=['pointnet', 'pointnet2',
                                 'dgcnn', 'pointconv','pct'],
                        help='Model to use, [pointnet, pointnet++, dgcnn, pointconv]')
    parser.add_argument('--lam_div', default=5., type=float, help='weight for div_loss')
    parser.add_argument('--lam_trans', default=5., type=float, help='weight for trans_loss')
    parser.add_argument('--lam_dis', default=10., type=float, help='weight for chamfer_dis')
    parser.add_argument('--z_dim', default=3, type=int, help='dimension of the latent vector')
    parser.add_argument('--epsilon', default=0.18, type=float, help='perturbation constraint')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--target_layer', type=str, default='dropout',
                    help='target layer : '
                         'dropout for pointnet,'
                         'drop2 for pointnet2,'
                         'dp2 for dgcnn,'
                         'drop2 for pointconv'
                         'dp2 for pct')
    parser.add_argument('--save_dir', default='mg_weight/', type=str, help='directory for saving model weights')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--sharply_root', type=str,
                        default='sharply_value/pointnet/pointnet_25players-concat.npz')
    parser.add_argument('--process_data', action='store_true', default=True, help='save data offline')
    args = parser.parse_args()
    BATCH_SIZE = BATCH_SIZE[args.num_point]
    BEST_WEIGHTS = BEST_WEIGHTS[args.dataset][args.num_point]
    if args.batch_size == -1:
        args.batch_size = BATCH_SIZE[args.model]
    set_seed(1)
    print(args)

    # dist.init_process_group(backend='nccl')
    # torch.cuda.set_device(args.local_rank)
    # enable cudnn benchmark
    cudnn.benchmark = True

    # build model
    if args.model.lower() == 'pct':
        model = Pct(args) # 模型输入[B,3,N]
    elif args.model.lower() == 'dgcnn':
        model = DGCNN(args.emb_dims, args.k, output_channels=40)
    elif args.model.lower() == 'pointnet':
        model = PointNetCls(k=40, feature_transform=args.feature_transform)
    elif args.model.lower() == 'pointnet2':
        model = PointNet2ClsSsg(num_classes=40)
    elif args.model.lower() == 'pointconv':
        model = PointConvDensityClsSsg(num_classes=40)
    else:
        print('Model not recognized')
        exit(-1)

    # model = nn.DataParallel(model).cuda()
        
    # load model weight
    state_dict = torch.load(
        BEST_WEIGHTS[args.model], map_location='cpu')
    print('Loading weight {}'.format(BEST_WEIGHTS[args.model]))

    if args.model.lower() == 'pointtransformer':
        # state_dict = {'module.'+k: v for k, v in state_dict['model_state_dict'].items()}
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            # eliminate 'module.' in keys
            state_dict = {k[7:]: v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)

    # distributed mode on multiple GPUs!
    # much faster than nn.DataParallel
    # if args.model.lower() != 'pointtransformer':
    # model = DistributedDataParallel(
    #             model.cuda(), device_ids=[args.local_rank], find_unused_parameters=True)
    model = model.cuda()
    # prepare data
    train_set = ModelNetDataLoader(root=args.data_root, args=args, split='train', process_data=args.process_data)
    test_set = ModelNetDataLoader(root=args.data_root, args=args, split='test', process_data=args.process_data)
    
    # train_sampler = DistributedSampler(train_set, shuffle=False)
    # test_sampler = DistributedSampler(test_set, shuffle=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=10, drop_last=True, sampler=None)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=10, drop_last=False, sampler=None)
    
    ### add methods
    if args.attack_methods=='3D-Adv':
         # setup attack settings
        if args.adv_func == 'logits':
            adv_func = UntargetedLogitsAdvLoss(kappa=args.kappa_3d_adv)
        else:
            adv_func = CrossEntropyAdvLoss()
        dist_func = L2Dist()
        clip_func = ClipPointsLinf(budget=args.epsilon)
        # hyper-parameters from their official tensorflow code
        attacker = CWPerturb(args.model.lower(), model, adv_func, dist_func,
                            attack_lr=args.attack_lr_3d_adv,
                            init_weight=10., max_weight=80.,
                            binary_step=args.binary_step_3d_adv,
                            num_iter=args.num_iter_3d_adv, clip_func=clip_func)
    
    elif args.attack_methods=='KNN':
         # setup attack settings
        if args.adv_func == 'logits':
            adv_func = UntargetedLogitsAdvLoss(kappa=args.kappa_knn)
        else:
            adv_func = CrossEntropyAdvLoss()
        # hyper-parameters from their official tensorflow code
        dist_func = ChamferkNNDist(chamfer_method='adv2ori',
                                knn_k=5, knn_alpha=1.05,
                                chamfer_weight=5., knn_weight=3.)
        clip_func = ProjectInnerClipLinf(budget=args.epsilon)
        attacker = CWUKNN(args.model.lower(),model, adv_func, dist_func, clip_func,
                        attack_lr=args.attack_lr_knn,
                        num_iter=args.num_iter_knn)
    
    elif args.attack_methods=='AdvPC':
         #AutoEncoder model
        ae_model = encoders_decoders.AutoEncoder(3)
        ae_state_dict = torch.load(args.ae_model_path, map_location='cpu')
        print('Loading ae weight {}'.format(args.ae_model_path))
        try:
            ae_model.load_state_dict(ae_state_dict)
        except RuntimeError:
            ae_state_dict = {k[7:]: v for k, v in ae_state_dict.items()}
            ae_model.load_state_dict(ae_state_dict)

        # ae_model = DistributedDataParallel(
        #     ae_model.cuda(), device_ids=[args.local_rank])
        ae_model = ae_model.cuda()
        # setup attack settings
        if args.adv_func == 'logits':
            adv_func = UntargetedLogitsAdvLoss(kappa=args.kappa_advpc)
        else:
            adv_func = CrossEntropyAdvLoss()
        clip_func = ClipPointsLinf(budget=args.epsilon)
        dist_func = ChamferDist()

        attacker = CWUAdvPC(model, ae_model, adv_func, dist_func,
                            attack_lr=args.attack_lr_advpc,
                            binary_step=args.binary_step_advpc,
                            num_iter=args.num_iter_advpc, GAMMA=args.GAMMA_advpc,
                            clip_func=clip_func)
    
    elif args.attack_methods=='PF-attack':
        # setup attack settings
        if args.adv_func_pf == 'logits':
            adv_func = UntargetedLogitsAdvLoss(kappa=args.kappa)
        else:
            adv_func = AdvCrossEntropyLoss()
        dist_func = ChamferDist()
        clip_func = ClipLinf(budget=args.epsilon)
        # hyper-parameters from their official tensorflow code
        attacker = CW_PFattack(model_name=args.model.lower(), model=model,  adv_func=adv_func,
                            dist_func=dist_func,players=args.players, 
                            k_sharp=args.k_sharp, initial_const = args.initial_const_pf,
                            pp = args.pp_pf, trans_weight = args.trans_weight_pf,
                            attack_lr=args.attack_lr_pf, binary_step=args.binary_step_pf,
                            num_iter=args.num_iter_pf, clip_func=clip_func)
    
    def attack(model, attacker, test_loader, point_index, second_noise, second_noise1, args):
        model.eval()
        ### generate attack results
        G = AutoEncoder_z(3)
        state_dict = torch.load(os.path.join(args.save_dir, args.model + '_eps_' +str(args.epsilon) + '.pth'),\
                                map_location='cpu')
        try:
            G.load_state_dict(state_dict)
        except RuntimeError:
            # eliminate 'module.' in keys
            state_dict = {k[7:]: v for k, v in state_dict.items()}
            G.load_state_dict(state_dict)
        print('Loading weight {}'.format(os.path.join(args.save_dir, args.model + '_eps_' +str(args.epsilon) + '.pth')))
        G = G.cuda()
    
        G.eval()
        pc, normal, label, sharply = iter(test_loader).__next__()#一次只输入一个数据点啊
        
        with torch.no_grad():
            pc, normal, label, sharply = pc.clone().float().cuda(non_blocking=True), normal.clone().float().cuda(non_blocking=True),\
            label.clone().long().cuda(non_blocking=True), sharply.float().cuda(non_blocking=True)
            double_points = torch.cat((pc, pc), dim=0)
            double_normal = torch.cat((normal, normal), dim=0)
            double_label = torch.cat((label, label), dim=0)

            single_points, single_normal, single_label, sharply = pc.clone().float().cuda(non_blocking=True), normal.clone().float().cuda(non_blocking=True),\
            label.clone().long().cuda(non_blocking=True), sharply.clone().float().cuda(non_blocking=True)
            adv_noise = torch.randn(pc.shape[0], pc.shape[1], 3).cuda() * 1e-7
        
        if args.finetune_mode=='noise':
            z = torch.FloatTensor(pc.shape[0] * 2, args.z_dim).normal_().cuda() #（B*2,16）
            width = double_points.shape[1]
            spatial_tile_z = torch.unsqueeze(z, -2).expand(-1, width, -1)
            double_adv_noise = G(double_points.transpose(1, 2).contiguous(), spatial_tile_z.transpose(1, 2).contiguous()) #相当于重构过程 forward大约3000M
        else:
            double_adv_noise = torch.randn(double_points.shape[0], double_points.shape[1], 3).cuda() * 1e-7
        
        # if point_index is not None:
        #     double_points = double_points[point_index:point_index+1]
        #     double_normal = double_normal[point_index:point_index+1]
        #     double_label = double_label[point_index:point_index+1]
        #     double_adv_noise = double_adv_noise[point_index:point_index+1]

        # if point_index is not None:
        #     single_points = single_points[point_index:point_index+1]
        #     single_normal = single_normal[point_index:point_index+1]
        #     single_label = single_label[point_index:point_index+1]
        #     adv_noise = adv_noise[point_index:point_index+1]
        
        if second_noise is not None:
            double_points = double_points + second_noise
        if second_noise1 is not None:
            single_points = single_points + second_noise1
        # attack!
        if args.attack_methods=='PF-attack':
            best_pc, success_num = attacker.attack(double_points, double_adv_noise, double_normal, double_label)
            single_best_pc, _ = attacker.attack(single_points, adv_noise, single_normal, single_label)
        else:
            best_pc, success_num = attacker.attack(double_points, double_adv_noise, double_label)
            single_best_pc, _ = attacker.attack(single_points, adv_noise, single_label)
        if best_pc.shape[1]!=1024:
            best_pc = best_pc.transpose(1,2)
            single_best_pc = single_best_pc.transpose(1,2)
        double_noise = best_pc-double_points
        single_noise = single_best_pc-single_points
        return double_noise, single_noise, single_points, single_label
    

    def loss_landscape(model, attacker, test_loader, point_index, args, z_axis_type="loss"):
    
        model.eval()

        double_noise, single_noise, single_points, single_label = attack(model, attacker, test_loader, point_index, None, None, args)
        double_noise1, single_noise1, _, _ = attack(model, attacker, test_loader, point_index, double_noise, single_noise, args)
        B = single_points.shape[0]
        first_axis = single_noise1 - single_noise
        second_axis = torch.randn_like(first_axis)
        # second_axis = single_noise1

        # Keep orthogonal direction
        second_axis -= (second_axis*first_axis).sum() / \
            (first_axis*first_axis).sum() * first_axis
        


        first_axis1 = double_noise1 - double_noise
        second_axis1 = torch.randn_like(first_axis1)
        # second_axis1 = double_noise1

        # Keep orthogonal direction
        second_axis1 -= (second_axis1*first_axis1).sum() / \
            (first_axis1*first_axis1).sum() * first_axis1

        axis_length = 100
        cross_ent = nn.CrossEntropyLoss()

        x_axis = np.linspace(-0.2, 0.2, axis_length)
        y_axis = np.linspace(-0.2, 0.2, axis_length)

        x = np.outer(x_axis, np.ones(axis_length))
        y = np.outer(y_axis, np.ones(axis_length)).T  # transpose
        z = np.zeros_like(x)
        z1 = np.zeros_like(x)
        colors = np.empty(x.shape, dtype=object)

        default_color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        default_color_cycle[7] = 'y'

        for i, j in tqdm(product(range(axis_length), range(axis_length))):
            current_point = (single_points + x_axis[i]*first_axis + y_axis[j]*second_axis)
            current_point = ClipPointsLinf(args.epsilon)(current_point, single_points)
            output = model(current_point.transpose(1, 2).contiguous())[0]
            # output of single image misses batch size
            output = output.view(-1, output.shape[-1])
            test_loss = cross_ent(output, single_label).item()
            if z_axis_type == "loss":
                z[i, j] = test_loss
            elif z_axis_type == "confidence":
                preds = nn.functional.softmax(output, dim=1)
                prediction = preds.argmax()
                z[i, j] = preds[0, single_label[0]].detach().cpu().numpy()
                colors[i, j] = default_color_cycle[prediction]
            else:
                raise NotImplementedError
            
        for i, j in tqdm(product(range(axis_length), range(axis_length))):
            current_point = (single_points + x_axis[i]*first_axis1[:B] + y_axis[j]*second_axis1[:B])
            current_point = ClipPointsLinf(args.epsilon)(current_point, single_points)
            output = model(current_point.transpose(1, 2).contiguous())[0]
            # output of single image misses batch size
            output = output.view(-1, output.shape[-1])
            test_loss = cross_ent(output, single_label).item()
            if z_axis_type == "loss":
                z1[i, j] = test_loss
            elif z_axis_type == "confidence":
                preds = nn.functional.softmax(output, dim=1)
                prediction = preds.argmax()
                z1[i, j] = preds[0, single_label[0]].detach().cpu().numpy()
                colors[i, j] = default_color_cycle[prediction]
            else:
                raise NotImplementedError


        fig = plt.figure()
        ax = plt.axes(projection='3d')

        if z_axis_type == "loss":
            ax.plot_surface(x, y, z, cmap='viridis', edgecolors='none')
            ax.set_title('Loss Landscape')
        elif z_axis_type == "confidence":
            ax.plot_surface(x, y, z, facecolors=colors,
                            edgecolors='none', shade=True)
            ax.set_title('Confidence Landscape')

        ax.set_xlim(np.min(x_axis), np.max(x_axis))
        ax.set_ylim(np.min(y_axis), np.max(y_axis))
        plt.ticklabel_format(axis='z', style='sci', scilimits=(0,0))
        ax.set_xlabel('adv')
        ax.set_ylabel(r'$adv^{\perp}$')
        ax.set_zlabel(z_axis_type)
        plt.savefig("%s_%s_3D.png" %(args.model, args.attack_methods), bbox_inches='tight')
        plt.savefig("%s_%s_3D.pdf" %(args.model, args.attack_methods), format='pdf', bbox_inches='tight')
        # 清除图形
        plt.clf()

        fig1 = plt.figure()
        ax1 = plt.axes(projection='3d')
        ax1.plot_surface(x, y, z1, cmap='viridis', edgecolors='none')
        ax1.set_title('Loss Landscape')
        ax1.set_xlim(np.min(x_axis), np.max(x_axis))
        ax1.set_ylim(np.min(y_axis), np.max(y_axis))
        plt.ticklabel_format(axis='z', style='sci', scilimits=(0,0))
        ax1.set_xlabel('adv')
        ax1.set_ylabel(r'$adv^{\perp}$')
        ax1.set_zlabel(z_axis_type)
        plt.savefig("ITAN_%s_%s_3D.png" %(args.model, args.attack_methods), bbox_inches='tight')
        plt.savefig("ITAN_%s_%s_3D.pdf" %(args.model, args.attack_methods), format='pdf', bbox_inches='tight')

        

    # for i in tqdm(range(1)):
    #     loss_landscape_args = dict(model=model,
    #                             attacker = attacker,
    #                             test_loader=test_loader,
    #                             point_index=i,
    #                             args = args,
    #                             z_axis_type="loss")

        # loss_landscape(**loss_landscape_args)
        
    def loss_landscape_(model, attacker, test_loader, point_index, type, args):
    
        model.eval()
        double_noise, single_noise, single_points, single_label = attack(model, attacker, test_loader, point_index, None, None, args)
        if type=='adv':
            double_noise1, single_noise1, _, _ = attack(model, attacker, test_loader, point_index, double_noise, single_noise, args)
        B = single_points.shape[0]
        x = single_points.clone() + single_noise
        x1 = single_noise + double_noise[:B]
        drawer = Landscape4Input(args, model, point_index, single_label, input=x.cuda(), input1=x1.cuda(), mode='2D', y_axis_='loss')
        drawer.synthesize_coordinates(x_min=-4, x_max=4, x_interval= 201) #应该就是0，18
        if type=='adv':
            direction_ = single_noise1
            direction_ /= torch.norm(direction_, p=float('inf'))
        else:
            direction = single_noise
            # direction /= torch.norm(direction, p=2)
            direction_random = torch.rand(direction.shape, device=direction.device)
            direction_random /= torch.norm(direction_random, p=float('inf'))

        
        # direction_1 /= torch.norm(direction_1, p=2)
        if type=='adv':
            direction1_ = double_noise1
            direction1_ /= torch.norm(direction1_, p=float('inf'))
        else:
            direction_1 = double_noise
            direction1_random = torch.rand(direction_1.shape, device=direction_1.device)
            direction1_random /= torch.norm(direction1_random, p=float('inf'))
        if type=='adv':
            drawer.assign_unit_vector(direction_, direction1_[:B])
            drawer.draw()
        else:
            drawer.assign_unit_vector(direction_random, direction1_random[:B])
            drawer.draw()

    
    for i in tqdm(range(100)):
        loss_landscape_(model, attacker, test_loader, i, 'adv', args)
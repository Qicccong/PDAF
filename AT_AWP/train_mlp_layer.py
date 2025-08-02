# RAWP 对抗训练过程


import argparse
from collections import defaultdict
import logging
import random
import sys
import time
import math
from bunch import Bunch
import evaluate_adv
from scipy.sparse import dok_matrix

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from mlp import MLP  
from Dataset import Dataset
# from utils import DataLoader

from wideresnet import WideResNet
from preactresnet import PreActResNet18
from evaluate import *
from utils_awp_layer import AdvWeightPerturb1

upper_limit, lower_limit = 1,0

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='MLP')
    parser.add_argument('--l2', default=0, type=float)
    parser.add_argument('--l1', default=0, type=float)
    parser.add_argument('--batch-size', default=200, type=int)
    parser.add_argument("--data_path", nargs="?", default="Data/",
                        help="Input data path.")
    parser.add_argument("--dataset", nargs="?", default="ml-1m",   # ml-1m  AMusic lastfm  #####################################
                        help="Choose a dataset.")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr-max', default=0.0001, type=float)        # 这块应该是推荐模型训练学习率 也可以 0.0005  ################################
    parser.add_argument('--lr-prox', default=0.01, type=float)   # 不动
    parser.add_argument('--fname', default='mlp_model', type=str)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--awp-gamma', default=0.001, type=float)   # 0.001,0.005   # 扰动参数的大小 ################################
    parser.add_argument("--alpha", nargs="?", default=0.005/10)  # 0.5/25
    parser.add_argument("--num_steps", nargs="?", default=10)  #25
    parser.add_argument("--adv_type", nargs="?", default="fgsm", choices=['fgsm', 'bim', 'pgd','mim'])   # 不动
    parser.add_argument("--decay_factor", nargs="?", default=1.0)
    parser.add_argument("--epsilon",  default=0.008, type=float)   # 检测鲁棒性 不动  008
    parser.add_argument("--pro_num", nargs="?", default=1, choices=[1, 25], help="1 for fgsm and 10 for bim/pgd")
    parser.add_argument('--awp-warmup', default=0, type=int)
    parser.add_argument("--fcLayers", nargs="?", default="[1024, 512, 256, 128, 64, 32, 16]", #  [512, 256, 128, 64, 32, 16]  [512,  128, 32]
                        help="Size of each layer. Note that the first layer is the "
                             "concatenation of user and item embeddings. So fcLayers[0]/2 is the embedding size.")
    parser.add_argument("--nNeg", type=int, default=4,help="Number of negative instances to pair with a positive instance.")
 
    return parser.parse_args()

def update_seed():
    seed = torch.rand(9)  # 生成六个介于0到1之间的随机数
    zs = [
        1 if x > 0.5 else 0  # 如果x大于0.5，返回1；否则返回0
        for x in seed
    ]
    return zs

def main():
    All_start_time = time.time()
    args = get_args()
    fcLayers = eval(args.fcLayers)
    topK = 10
    topK1 = 20
    topK2 = 50
    topK3 = 100
    
    # 把参数转成bunch
    args_model = Bunch()
    for key, value in args.__dict__.items():
        setattr(args_model, key, value)
    for key, value in args_model.items():
        print(f"{key}: {value}")
    print('--'*50)
    print('gamma不调整,每轮一次调整加噪层次')


    if args.awp_gamma <= 0.0:
        args.awp_warmup = np.inf

    if not os.path.exists(args.fname):
        os.makedirs(args.fname)

    logger = logging.getLogger(__name__)    # 这段代码的作用是配置一个具有指定格式和级别的日志记录器，以便记录程序执行时的相关信息。文件和控制台都被用作输出目标。
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.fname, 'eval.log' if args.eval else 'output.log')),
            logging.StreamHandler()
        ])

    logger.info(args)

    # Loading data
    t1 = time.time()
    dataset = Dataset(args.data_path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    nUsers, nItems = train.shape
    
    arr = np.array(testRatings)
    max_value_second_column1 = np.max(arr[:, 1])
    arr = np.array(testNegatives)
    max_value_second_column2 = np.max(arr)
    n_value = max(max_value_second_column1, max_value_second_column2)

    if nItems <= n_value:
        nItems = n_value+1
        extended_sparse_matrix = dok_matrix((nUsers, nItems), dtype=np.float32)

        for (row, col), value in train.items():
            extended_sparse_matrix[row, col] = value
            train = extended_sparse_matrix



    userMatrix = torch.Tensor(get_train_matrix(train))
    # train_batches = Batches(userMatrix, args.batch_size, shuffle=True, set_random_choices=True, num_workers=2)
    
    itemMatrix = torch.transpose(torch.Tensor(get_train_matrix(train)), 0, 1)
    userMatrix, itemMatrix = userMatrix.to(device), itemMatrix.to(device)

    print(f"Load data: #user={nUsers}, #item={nItems}, #train={train.nnz}, #test={len(testRatings)} [{time.time()-t1:.1f}s]")

    print("MLP_adv:*************************************")
    if args.model == 'MLP':
        model_adv = MLP(fcLayers, userMatrix, itemMatrix,device)
        proxy_adv = MLP(fcLayers, userMatrix, itemMatrix,device)
    else:
        raise ValueError("Unknown model")

    model_adv = nn.DataParallel(model_adv).to(device)
    proxy_adv = nn.DataParallel(proxy_adv).to(device)

    if args.l2:
        decay, no_decay = [], []
        for name,param in model_adv.named_parameters():
            if 'bn' not in name and 'bias' not in name:
                decay.append(param)
            else:
                no_decay.append(param)
        params = [{'params':decay, 'weight_decay':args.l2},
                  {'params':no_decay, 'weight_decay': 0 }]
    else:
        params = model_adv.parameters()

    # opt = torch.optim.SGD(params, lr=args.lr_max, momentum=0.9, weight_decay=5e-4)
    opt = torch.optim.Adam(params, lr=args.lr_max)
    proxy_opt = torch.optim.Adam(proxy_adv.parameters(), lr=args.lr_prox)   # 0.01
    awp_adversary = AdvWeightPerturb1(model=model_adv, proxy=proxy_adv, proxy_optim=proxy_opt, gamma=args.awp_gamma)
    criterion = torch.nn.BCELoss()
    

    # 不被攻击时的性能
    hits10, ndcgs10, maps10, mrrs10, hits20, ndcgs20,maps20, mrrs20,hits50, ndcgs50,maps50, mrrs50 = evaluate_model(model_adv, testRatings, testNegatives, topK,topK1,topK2,topK3, num_thread=1, device=device)
    hr10, ndcg10,map10,mrr10, hr20, ndcg20,map20,mrr20,hr50, ndcg50,map50, mrr50 = np.array(hits10).mean(), np.array(ndcgs10).mean(),np.array(maps10).mean(), np.array(mrrs10).mean(),\
        np.array(hits20).mean(), np.array(ndcgs20).mean(), np.array(maps20).mean(), np.array(mrrs20).mean(), np.array(hits50).mean(), np.array(ndcgs50).mean(), np.array(maps50).mean(), np.array(mrrs50).mean()
    print(f"init: HR10={hr10:.4f}, NDCG10={ndcg10:.4f}, mrrs10={mrr10:.4f}, HR20={hr20:.4f}, NDCG20={ndcg20:.4f}, mrrs20={mrr20:.4f}, HR50={hr50:.4f}, NDCG50={ndcg50:.4f}, mrrs50={mrr50:.4f} [{time.time()-t1:.1f}s]")
        
    
    for epoch in range(args.epochs):
        start_time = time.time()
        # Generate training instances
        userInput, itemInput, labels = get_train_instances(train, args.nNeg)

        dst = BatchDataset(userInput, itemInput, labels)
        ldr = torch.utils.data.DataLoader(dst, batch_size=args.batch_size, shuffle=True, drop_last=True)
        losses = AverageMeter("Loss")
        
        if epoch>=0:
            # args.awp_gamma, args.pro_num = set_noise(epoch,adv_type=args.adv_type)
            # print(args.awp_gamma, args.pro_num)
            T = random.uniform(0.8, 1.2)
            args.awp_gamma_temp = args.awp_gamma*T
            # args.awp_gamma_temp = args.awp_gamma
            awp_adversary = AdvWeightPerturb1(model=model_adv, proxy=proxy_adv, proxy_optim=proxy_opt, gamma=args.awp_gamma_temp)
        
        seeds = update_seed()
        print(seeds, args.awp_gamma_temp, args.pro_num)
        for ui, ii, lbl in ldr:
            if args.eval:
                break
            ui, ii, lbl = ui.to(device), ii.to(device), lbl.to(device)  
            model_adv.train()
            # calculate adversarial weight perturbation and perturb it
            if epoch >= args.awp_warmup:
                awp = awp_adversary.calc_awp(inputs_adv_u=ui, inputs_adv_i=ii,inputs_adv_lab=lbl,
                                             attack_method=args.adv_type,pro_num=args.pro_num,)
                awp_adversary.perturb(awp,seeds)

            # robust_output,_ = model_adv(ui, ii)
            robust_loss,_,_,_,_,_,_,_= model_adv(ui, ii,lbl)
            # robust_loss = criterion(robust_output, lbl)

            if args.l1:
                for name,param in model_adv.named_parameters():
                    if 'bn' not in name and 'bias' not in name:
                        robust_loss += args.l1*param.abs().sum()

            opt.zero_grad()
            robust_loss.backward()
            opt.step()
            losses.update(robust_loss.item(), lbl.size(0))
            if epoch >= args.awp_warmup:
                awp_adversary.restore(awp,seeds)

            # output,_ = model_adv(ui, ii)
            # loss = criterion(output, lbl)

        model_adv.eval()
        # modelPath = f"pretrained/{args.dataset}_MLP_adv_{time.time()}.pth"
        modelPath = f"pretrained/{args.dataset}_MLP_init_{time.time()}.pth"
        
        # Check performance
        t1 = time.time()
        hits10, ndcgs10, maps10, mrrs10, hits20, ndcgs20,maps20, mrrs20,hits50, ndcgs50,maps50, mrrs50 = evaluate_model(model_adv, testRatings, testNegatives, topK,topK1,topK2,topK3, num_thread=1, device=device)
        hr10, ndcg10,map10,mrr10, hr20, ndcg20,map20,mrr20,hr50, ndcg50,map50, mrr50 = np.array(hits10).mean(), np.array(ndcgs10).mean(),np.array(maps10).mean(), np.array(mrrs10).mean(),\
            np.array(hits20).mean(), np.array(ndcgs20).mean(), np.array(maps20).mean(), np.array(mrrs20).mean(), np.array(hits50).mean(), np.array(ndcgs50).mean(), np.array(maps50).mean(), np.array(mrrs50).mean()
        if epoch == 0 :
            bestHr10, bestNdcg10,bestmap10, bestmrr10,bestHr20, bestNdcg20,bestmap20, bestmrr20,bestHr50, bestNdcg50,bestmap50, bestmrr50, bestEpoch = hr10, ndcg10,map10,mrr10, hr20, ndcg20,map20,mrr20,hr50, ndcg50,map50, mrr50 , -1
    
        logger.info(f"Epoch {epoch+1}:Loss={losses.avg:.4f} [{t1-start_time:.1f}s] HR10={hr10:.4f}, NDCG10={ndcg10:.4f}, mrrs10={mrr10:.4f}, HR20={hr20:.4f}, NDCG20={ndcg20:.4f}, mrrs20={mrr20:.4f}, HR50={hr50:.4f}, NDCG50={ndcg50:.4f}, mrrs50={mrr50:.4f} [{time.time()-t1:.1f}s]")
        
        hits10, ndcgs10, maps10, mrrs10, hits20, ndcgs20,maps20, mrrs20,hits50, ndcgs50,maps50, mrrs50 = evaluate_adv.evaluate_model(model_adv, testRatings, testNegatives, topK,topK1,topK2,topK3,args.adv_type,args.epsilon,args.alpha,args.num_steps, args.decay_factor, num_thread=1, device=device)
        # evaluate_model(model, testRatings, testNegatives, K, K1, K2,K3,attack1,epsilon1,alpha1,num_steps1,decay_factor, num_thread, device)
        # print('hit10_robust:{}'.format(hit10_robust))
        hr10_R, ndcg10,map10,mrr10, hr20, ndcg20,map20,mrr20,hr50, ndcg50,map50, mrr50 = np.array(hits10).mean(), np.array(ndcgs10).mean(),np.array(maps10).mean(), np.array(mrrs10).mean(),\
        np.array(hits20).mean(), np.array(ndcgs20).mean(), np.array(maps20).mean(), np.array(mrrs20).mean(), np.array(hits50).mean(), np.array(ndcgs50).mean(), np.array(maps50).mean(), np.array(mrrs50).mean()
        logger.info(f"Robust: HR10={hr10_R:.4f}, NDCG10={ndcg10:.4f}, mrrs10={mrr10:.4f}, HR20={hr20:.4f}, NDCG20={ndcg20:.4f}, mrrs20={mrr20:.4f}, HR50={hr50:.4f}, NDCG50={ndcg50:.4f}, mrrs50={mrr50:.4f} [{time.time()-t1:.1f}s]")
    
        # 
        new_hr = hr10 + hr10_R
        if new_hr > bestHr10:
            bestHr10 = hr10 + hr10_R
            os.makedirs("pretrained", exist_ok=True)
            torch.save(model_adv.state_dict(), modelPath)
            logger.info("save:=================================")
            logger.info(modelPath)
    
    print(time.time()-All_start_time)
    


if __name__ == "__main__":
    main()

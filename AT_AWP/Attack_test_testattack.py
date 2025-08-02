import argparse
import logging
import sys
import time
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from mlp import MLP
from Dataset import Dataset
# from utils import DataLoader

from scipy.sparse import dok_matrix
from wideresnet import WideResNet
from preactresnet import PreActResNet18
from evaluate_adv import *
from utils_awp import AdvWeightPerturb1



upper_limit, lower_limit = 1,0

EPS = 1E-20

def cal_lp_norm(tensor,p,dim_count):
    tmp = tensor
    for i in range(1,dim_count):
        tmp = torch.norm(tmp,p=p,dim=i,keepdim=True) #torch.Size([100, 1])
    
    
    return torch.clamp_min(tmp, 1e-8)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='MLP')
    parser.add_argument('--l2', default=0, type=float)
    parser.add_argument('--l1', default=0, type=float)
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument("--data_path", nargs="?", default="Data/",
                        help="Input data path.")
    parser.add_argument("--dataset", nargs="?", default="AMusic",   # ml-1m yelp  AToy lastfm
                        help="Choose a dataset.")
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr-max', default=0.0005, type=float)        # 0.1 这块应该是推荐模型训练学习率 RAT ml1m 0.0005
    parser.add_argument('--lr-prox', default=0.05, type=float)
    parser.add_argument('--attack', default='fgsm', type=str, choices=['fgsm', 'fgsm', 'bim', 'none' ,'Gaussian ','Uniform ', 'pgd', 'mim'])
    parser.add_argument('--epsilon', default=0.008 ,type=float) 
    parser.add_argument("--alpha", type=float ,default=0.008/10)  # 0.5/25  nargs="?",
    parser.add_argument("--num_steps", type=int, default=10)  #25
    # parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'])
    parser.add_argument('--fname', default='mlp_model', type=str)
    # parser.add_argument('--seed', default=123, type=int)
    # parser.add_argument('--half', action='store_true')
    # parser.add_argument('--width-factor', default=10, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--val', action='store_true')
    parser.add_argument("--decay_factor", nargs="?", default=1.0)
    # parser.add_argument('--chkpt-iters', default=10, type=int)
    # parser.add_argument('--awp-gamma', default=0.00001, type=float)
    # parser.add_argument('--awp-warmup', default=0, type=int)
    parser.add_argument("--fcLayers", nargs="?", default="[1024, 512, 256, 128, 64, 32, 16]", #  [512, 256, 128, 64, 32, 16]  [512,  128, 32]
                        help="Size of each layer. Note that the first layer is the "
                             "concatenation of user and item embeddings. So fcLayers[0]/2 is the embedding size.")
    parser.add_argument("--nNeg", type=int, default=4,help="Number of negative instances to pair with a positive instance.")
    parser.add_argument("--model_path", type=str, default="",help="choose a parameter file")
 
    return parser.parse_args()


def main():
    All_start_time = time.time()
    args = get_args()
    fcLayers = eval(args.fcLayers)
    topK = 10
    topK1 = 20
    topK2 = 50
    topK3 = 100
    print("比例l2,每一轮改变模型的参数都不变,不累积影响下去")
    print('--'*50)


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

    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    
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

    itemMatrix = torch.transpose(torch.Tensor(get_train_matrix(train)), 0, 1)
    userMatrix, itemMatrix = userMatrix.to(device), itemMatrix.to(device)
    print(f"Load data: #user={nUsers}, #item={nItems}, #train={train.nnz}, #test={len(testRatings)} [{time.time()-t1:.1f}s]")
    print("MLP_Attack_test:*************************************")
    if args.model == 'MLP':
        model_adv = MLP(fcLayers, userMatrix, itemMatrix,device)
    else:
        raise ValueError("Unknown model")

    model_adv = nn.DataParallel(model_adv).to(device)

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

    model_path = args.model_path
    # model_path = 'AT_AWP\\param_file\\RAWP\\79_lastfm_MLP_RAWP_1718355934.5297291.pth'

    print(model_path)
    print("attack_weight")
    # if 'robust'or 'MLP' in model_path: 
    if 'robust' in model_path:  
        # 假设 original_state_dict 是你加载的原始状态字典 
        original_state_dict = torch.load(model_path,map_location=device)
        # # 如果是mlp生成的模型，有net需要提取出来
        # original_state_dict = original_state_dict['net']
        
        # 修改键以匹配 DataParallel 的期望格式
        modified_state_dict = {'module.' + key: value for key, value in original_state_dict.items()}
        modified_state_dict = original_state_dict

        # 创建一个新的state_dict，只包含不带有'reg'字段的参数
        new_state_dict111 = {}
        for key, value in modified_state_dict.items():
            if 'reg' not in key:
                new_state_dict111[key] = value
        model_adv.load_state_dict(new_state_dict111)
        
        # # 现在使用修改后的状态字典加载模型
        # model_adv.load_state_dict(modified_state_dict) 
    else:  # RAWP
        original_state_dict = torch.load(model_path,map_location=device)
        # 创建一个新的state_dict，只包含不带有'reg'字段的参数
        new_state_dict111 = {}
        for key, value in original_state_dict.items():
            if 'reg' not in key:
                new_state_dict111[key] = value
        model_adv.load_state_dict(new_state_dict111)
        
    del new_state_dict111
    torch.cuda.empty_cache()

    # Check  performance
    t1 = time.time()
    hits10, ndcgs10, maps10, mrrs10 = evaluate_model(model_adv, testRatings, testNegatives, topK,topK1,topK2,topK3,args.attack,args.epsilon,args.alpha,args.num_steps, args.decay_factor, num_thread=1, device=device)
    hr10, ndcg10,map10,mrr10 = np.array(hits10).mean(), np.array(ndcgs10).mean(),np.array(maps10).mean(), np.array(mrrs10).mean()
    print(f"Init: HR10={hr10:.4f}, NDCG10={ndcg10:.4f}, mrrs10={mrr10:.4f}")
    
    
    

if __name__ == "__main__":
    main()

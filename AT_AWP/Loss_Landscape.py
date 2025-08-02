import argparse
import logging
import sys
import time
import math
from bunch import Bunch
from matplotlib import pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os
from mlp_test import MLP
# from mlp import MLP
from Dataset import Dataset
# from utils import DataLoader

# from wideresnet import WideResNet
# from preactresnet import PreActResNet18
from evaluate import *
# from utils_awp import AdvWeightPerturb1

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='MLP')
    parser.add_argument('--l2', default=0, type=float)
    parser.add_argument('--l1', default=0, type=float)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument("--data_path", nargs="?", default="Data/",
                        help="Input data path.")
    parser.add_argument("--dataset", nargs="?", default="lastfm",   # ml-1m yelp  AToy lastfm
                        help="Choose a dataset.")
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr-max', default=0.05, type=float)        # 0.1 这块应该是推荐模型训练学习率 RAT ml1m 0.0005
    parser.add_argument('--lr-prox', default=0.01, type=float)
    parser.add_argument('--attack', default='bim', type=str, choices=['pgd', 'fgsm', 'free', 'none'])
    parser.add_argument('--epsilon', default=0.5, type=int)
    parser.add_argument("--alpha", nargs="?", default=0.5/25)  # 0.5/25
    parser.add_argument("--num_steps", nargs="?", default=25)  #25
    # parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'])
    parser.add_argument('--fname', default='mlp_model', type=str)
    parser.add_argument('--seed', default=1, type=int)
    # parser.add_argument('--half', action='store_true')
    # parser.add_argument('--width-factor', default=10, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--val', action='store_true')
    # parser.add_argument('--chkpt-iters', default=10, type=int)
    parser.add_argument('--awp-gamma', default=0.00001, type=float)
    parser.add_argument('--awp-warmup', default=0, type=int)
    parser.add_argument("--fcLayers", nargs="?", default="[1024, 512, 256, 128, 64, 32, 16]", #  [512, 256, 128, 64, 32, 16]  [512,  128, 32]
                        help="Size of each layer. Note that the first layer is the "
                             "concatenation of user and item embeddings. So fcLayers[0]/2 is the embedding size.")
    parser.add_argument("--nNeg", type=int, default=4,help="Number of negative instances to pair with a positive instance.")
 
    return parser.parse_args()


def main():
    All_start_time = time.time()
    args = get_args()
    fcLayers = eval(args.fcLayers)
    topK = 10
    topK1 = 20
    topK2 = 50
    topK3 = 100
    # 微调的步长和范围
    steps = 100
    delta = 0.04
    rounds = 10
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    # Loading data
    t1 = time.time()
    dataset = Dataset(args.data_path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    nUsers, nItems = train.shape
    userMatrix = torch.Tensor(get_train_matrix(train))
    
    itemMatrix = torch.transpose(torch.Tensor(get_train_matrix(train)), 0, 1)
    userMatrix, itemMatrix = userMatrix.to(device), itemMatrix.to(device)
    print(f"Load data: #user={nUsers}, #item={nItems}, #train={train.nnz}, #test={len(testRatings)} [{time.time()-t1:.1f}s]")
    print("MLP_Attack_test:*************************************")
    if args.model == 'MLP':
        model_adv1 = MLP(fcLayers, userMatrix, itemMatrix,device)
    else:
        raise ValueError("Unknown model")

    model_adv1 = nn.DataParallel(model_adv1).to(device)

    criterion = torch.nn.BCELoss()

    # 加载预训练的参数 
    # model_path1 = 'clean_lastfm_1e-05_mim_MLP_robust.pth'
    # model_path1 = '/mnt/home/E22301274/Def/AWP-main_new01/64_lastfm_MLP_clean_8840_1710387934.3992233.pth'
    # model_path1 = "AT_AWP\param_file\RAWP\80_ml-1m_MLP_init_1713513910.3174512.pth"  #RAWP ml-1m
    model_path1 = "AT_AWP/param_file/RAWP/79_lastfm_MLP_RAWP_1718355934.5297291.pth"
    print(model_path1)
    # model_path2 = 'lastfm_1e-05_mim_MLP_robust_eval.pth'
    # model_path2 = "AT_AWP\param_file\RAWP-FT\ml-1m_RAWP_80_FineTurning_1715970106.2888427.pth"
    model_path2 = "AT_AWP/param_file/RAWP-FT/lastfm_RAWP_79_FineTurning_1718535631.2096848.pth"
    print(model_path2)
    # model_path3 = 'lastfm_MLP_adv_1709210611.3623106.pth'
    # print(model_path3)
    # model_path3 = '58_lastfm_MLP_adv_1709210611.3623106.pth'
    # model_path3 = "AT_AWP\param_file\FTS\ML-1M\ml-1m_FTS_lr-0.005_gamma-0.001.pth"
    model_path3 = "AT_AWP/param_file/FTS/lastfm/lastfm_FTS_lr-0.05_gamma-0.008.pth"
    print(model_path3)
    
    
    userInput, itemInput, labels = get_train_instances(train, args.nNeg)
    # dst = BatchDataset(userInput, itemInput, labels)
    dst = BatchDataset([userInput[109]], [itemInput[109]], [labels[109]]) 
    ldr = torch.utils.data.DataLoader(dst, batch_size=args.batch_size, shuffle=True, drop_last=True)


    model_paths = [model_path1, model_path2, model_path3]
    model_labels = ['clean_8857', 'RAT_8754', 'RAWP_8845']
    losses_avg = np.zeros((len(model_paths), 2*steps+1))
    loss_functions = [adjust_and_record_loss, adjust_and_record_loss, adjust_and_record_loss_RAWP]
    for round_num in range(rounds):
        directions = []
        for p in model_adv1.parameters():
            # 从高斯分布中为p的形状采样
            d = torch.randn_like(p)
            # 对采样得到的方向进行Frobenius范数归一化
            norm = torch.norm(d, p=2)
            d_normalized = d / (norm + 1e-10)  # 添加小的常数防止除以零
            directions.append(d_normalized)  #86705
        
        for i, (model_path, label,loss_function) in enumerate(zip(model_paths, model_labels,loss_functions)):
            losses_model = loss_function(round_num,model_adv1, model_path, ldr, directions, criterion, testRatings, testNegatives, topK, topK1, topK2, topK3, steps, delta)
            losses_avg[i] += np.array(losses_model)
        print(time.time() - t1)
        
    losses_avg /= rounds
    x_axis = np.linspace(-steps*delta, steps*delta, 2*steps+1)

    # 绘制损失Landscape
    for i, label in enumerate(model_labels):
        plt.plot(x_axis, losses_avg[i], label=label)
    
    plt.xlabel('Weight adjustment in the direction')
    plt.ylabel('Loss')
    plt.title('Loss Landscape of MLP Recommendation Model')
    plt.legend()
    path_save = f'loss_landscape_1D_lastfm_rounds:{rounds}_{steps}_{delta}_{time.time()}.png'
    plt.savefig(path_save)
    print(path_save)
    

def adjust_and_record_loss(ii,model,model_path, ldr, directions,criterion,testRatings, testNegatives, topK,topK1,topK2,topK3,steps=50, delta=0.01):
    # 加载预训练的参数 
    # model_path = 'clean_lastfm_1e-05_mim_MLP_robust.pth'
    print(model_path)
    # 假设 original_state_dict 是你加载的原始状态字典
    original_state_dict = torch.load(model_path, map_location=device)
    # 修改键以匹配 DataParallel 的期望格式
    # modified_state_dict = {'module.' + key: value for key, value in original_state_dict.items()}
    # modified_state_dict = {key.replace('module.', '', 1): value for key, value in original_state_dict.items()}
    # 现在使用修改后的状态字典加载模型
    # model.load_state_dict(modified_state_dict) 
    model.load_state_dict(original_state_dict, strict=False)
    model.eval()
    # if ii == 0:
    #     # Check  performance
    #     t1 = time.time()
    #     hits10, ndcgs10, maps10, mrrs10, hits20, ndcgs20,maps20, mrrs20,hits50, ndcgs50,maps50, mrrs50 = evaluate_model(model, testRatings, testNegatives, topK,topK1,topK2,topK3, num_thread=1, device=device)
    #     hr10, ndcg10,map10,mrr10, hr20, ndcg20,map20,mrr20,hr50, ndcg50,map50, mrr50 = np.array(hits10).mean(), np.array(ndcgs10).mean(),np.array(maps10).mean(), np.array(mrrs10).mean(),\
    #         np.array(hits20).mean(), np.array(ndcgs20).mean(), np.array(maps20).mean(), np.array(mrrs20).mean(), np.array(hits50).mean(), np.array(ndcgs50).mean(), np.array(maps50).mean(), np.array(mrrs50).mean()
    #     print(f"model Init: HR10={hr10:.4f}, NDCG10={ndcg10:.4f}, mrrs10={mrr10:.4f}, HR20={hr20:.4f}, NDCG20={ndcg20:.4f}, mrrs20={mrr20:.4f}, HR50={hr50:.4f}, NDCG50={ndcg50:.4f}, mrrs50={mrr50:.4f} [{time.time()-t1:.1f}s]")
    
    # 计算损失
    losses = []
    for step in range(-steps, steps+1):
        # 保存原始参数
        original_params = [p.data.clone() for p in model.parameters()]

        # 在选定的方向上微调权重
        for p, d in zip(model.parameters(), directions):
            p.data += d * delta * step

        # 计算损失
        loss_sum = 0
        for ui, ii, lbl in ldr:
            ui, ii, lbl = ui.to(device), ii.to(device), lbl.to(device)  
            loss,_,_,_,_,_,_,output= model(ui, ii,lbl) ##########减去原始损失！加了PGD之后得到损失减去没有加PGD的clean下的损失
            # yc = output.squeeze()
            # loss = criterion(yc, lbl)
            loss_sum += loss.item()
        losses.append(loss_sum / len(ldr))
        if(step==0):
            adjusted_loss = loss_sum / len(ldr)

        # 恢复原始参数
        for p, original in zip(model.parameters(), original_params):
            p.data = original
            
    # for i in range(len(losses)):  # 减去0处损失
    #     losses[i]=losses[i]-adjusted_loss 

    return losses

def adjust_and_record_loss_RAWP(ii,model,model_path, ldr, directions,criterion,testRatings, testNegatives, topK,topK1,topK2,topK3,steps=50, delta=0.01):
    # 加载预训练的参数 
    # # model_path = 'clean_lastfm_1e-05_mim_MLP_robust.pth'
    print(model_path)
    model.load_state_dict(torch.load(model_path)) 
    model.eval()
    # if ii == 0:
    #     # Check  performance
    #     t1 = time.time()
    #     hits10, ndcgs10, maps10, mrrs10, hits20, ndcgs20,maps20, mrrs20,hits50, ndcgs50,maps50, mrrs50 = evaluate_model(model, testRatings, testNegatives, topK,topK1,topK2,topK3, num_thread=1, device=device)
    #     hr10, ndcg10,map10,mrr10, hr20, ndcg20,map20,mrr20,hr50, ndcg50,map50, mrr50 = np.array(hits10).mean(), np.array(ndcgs10).mean(),np.array(maps10).mean(), np.array(mrrs10).mean(),\
    #         np.array(hits20).mean(), np.array(ndcgs20).mean(), np.array(maps20).mean(), np.array(mrrs20).mean(), np.array(hits50).mean(), np.array(ndcgs50).mean(), np.array(maps50).mean(), np.array(mrrs50).mean()
    #     print(f"model Init: HR10={hr10:.4f}, NDCG10={ndcg10:.4f}, mrrs10={mrr10:.4f}, HR20={hr20:.4f}, NDCG20={ndcg20:.4f}, mrrs20={mrr20:.4f}, HR50={hr50:.4f}, NDCG50={ndcg50:.4f}, mrrs50={mrr50:.4f} [{time.time()-t1:.1f}s]")
    
    # 计算损失
    losses = []
    for step in range(-steps, steps+1):
        # 保存原始参数
        original_params = [p.data.clone() for p in model.parameters()]

        # 在选定的方向上微调权重
        for p, d in zip(model.parameters(), directions):
            p.data += d * delta * step

        # 计算损失
        loss_sum = 0
        for ui, ii, lbl in ldr:  ###一次只用一个数据点，你这个把整个数据集都用进去了啊
            ui, ii, lbl = ui.to(device), ii.to(device), lbl.to(device)  
            # model.train()
            loss,_,_,_,_,_,_,output= model(ui, ii,lbl)
            # yc = output.squeeze()
            # loss = criterion(yc, lbl)
            loss_sum += loss.item()
        losses.append(loss_sum / len(ldr))
        if(step==0):
            adjusted_loss = loss_sum / len(ldr)

        # 恢复原始参数
        for p, original in zip(model.parameters(), original_params):
            p.data = original
    for i in range(len(losses)):
        losses[i]=losses[i]-adjusted_loss
    return losses


if __name__ == "__main__":
    main()

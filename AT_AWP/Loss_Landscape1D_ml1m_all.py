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
from Dataset import Dataset
# from utils import DataLoader

# from wideresnet import WideResNet
# from preactresnet import PreActResNet18
from AT_AWP.evaluate import *
# from utils_awp import AdvWeightPerturb1

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='MLP')
    parser.add_argument('--l2', default=0, type=float)
    parser.add_argument('--l1', default=0, type=float)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument("--data_path", nargs="?", default="Data/",
                        help="Input data path.")
    parser.add_argument("--dataset", nargs="?", default="ml-1m",   # ml-1m yelp  AToy lastfm
                        help="Choose a dataset.")
    # parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'])
    parser.add_argument('--fname', default='mlp_model', type=str)
    parser.add_argument('--seed', default=123, type=int)
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
    delta = 0.02
    rounds = 200
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

    userInput, itemInput, labels = get_train_instances(train, args.nNeg)
    dst = BatchDataset(userInput, itemInput, labels)
    # dst = BatchDataset([0], [32], [1]) 
    ldr = torch.utils.data.DataLoader(dst, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    
    # # 初始化三个空数组用于存储用户ID、项目ID和标签
    # user_ids = []
    # project_ids = []
    # labels = []
    # # 处理交互数组
    # for user_id, row in enumerate(testNegatives):
    #     for idd, project_id in enumerate(row):
    #         # if interaction != 0:
    #         user_ids.append(user_id)
    #         project_ids.append(project_id)
    #         labels.append(1)
    # dst = BatchDataset(user_ids, project_ids, labels)
    # ldr = torch.utils.data.DataLoader(dst, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    # 加载预训练的参数 
    model_path1 = '29_ml-1m_0.0005_mim_MLP_clean.pth'
    model_path2 = '28_7030_ml-1m_1e-05_mim_MLP_robust.pth'
    model_path3 = '30_1_RAT_ml-1m_0.0005_fgsm_MLP_robust.pth'
    model_path4 = '41_ml-1m_MLP_adv_1710225337.0569274.pth'
    model_path5 = '43_ml1m_RAWP_ml-1m_MLP_adv_1710309496.812673.pth'
    model_path6 = '27_ml-1m_RAWP_adv_1709724672.2920473.pth'
    # model_path7 = '42_ml-1m_1e-05_pgd_MLP_robust.pth'
    model_path8 = '46_ml-1m_MLP_adv_1710940578.3637655.pth'
    print(model_path1)
    print(model_path2)
    print(model_path3)
    print(model_path4)
    model_paths = [model_path1, model_path2,model_path3,model_path4,model_path5,model_path6,model_path8]
    model_labels = ['clean_29', 'RAT_28_mim', 'RAT_30_1_fgsm','41RAWP','43RAWP','27_1_RAWP','46RAWP']
    losses_avg = np.zeros((len(model_paths), 2*steps+1))
    loss_functions = [adjust_and_record_loss, adjust_and_record_loss,adjust_and_record_loss, adjust_and_record_loss_RAWP,adjust_and_record_loss_RAWP,adjust_and_record_loss_RAWP
                      ,adjust_and_record_loss_RAWP,adjust_and_record_loss,adjust_and_record_loss_RAWP,adjust_and_record_loss_RAWP]
    
    ui, ii, lbl = iter(ldr).__next__()
    ui, ii, lbl = ui.to(device), ii.to(device), lbl.to(device)
    for round_num in range(rounds):
        directions = []
        for p in model_adv1.parameters():
            # 从高斯分布中为p的形状采样
            d = torch.randn_like(p)
            # 对采样得到的方向进行Frobenius范数归一化
            norm = torch.norm(d, p=2)
            d_normalized = d / (norm + 1e-10)  # 添加小的常数防止除以零
            directions.append(d_normalized)  
        
        for i, (model_path, label,loss_function) in enumerate(zip(model_paths, model_labels,loss_functions)):
            losses_model = loss_function(round_num,model_adv1, model_path, ui, ii, lbl, directions, criterion, testRatings, testNegatives, topK, topK1, topK2, topK3, steps, delta)
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
    path_save = f'loss_train_landscape_1D_ml1m_bsz{args.batch_size}seed{args.seed}rounds:{rounds}_{steps}_{delta}_{time.time()}.png'
    plt.savefig(path_save)
    print(path_save)
    

def adjust_and_record_loss(ii11,model,model_path, ui, ii, lbl, ldr,directions,criterion,testRatings, testNegatives, topK,topK1,topK2,topK3,steps=50, delta=0.01):
    # 加载预训练的参数 
    # print(model_path)
    # 假设 original_state_dict 是你加载的原始状态字典
    original_state_dict = torch.load(model_path)
    # 修改键以匹配 DataParallel 的期望格式
    modified_state_dict = {'module.' + key: value for key, value in original_state_dict.items()}
    # 现在使用修改后的状态字典加载模型
    model.load_state_dict(modified_state_dict) 
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
    # ui, ii, lbl = iter(ldr).__next__()
    # ui, ii, lbl = ui.to(device), ii.to(device), lbl.to(device)
    for step in range(-steps, steps+1):
        # 保存原始参数
        original_params = [p.data.clone() for p in model.parameters()]

        # 在选定的方向上微调权重
        for p, d in zip(model.parameters(), directions):
            p.data += d * delta * step

        # 计算损失
        loss_sum = 0
        # model.train()
        # loss,_,_,_,_,_,_,output= model(ui, ii,lbl)
        # # yc = output.squeeze()
        # # loss = criterion(yc, lbl)
        # loss_sum += loss.item()
        for ui, ii, lbl in ldr:
            ui, ii, lbl = ui.to(device), ii.to(device), lbl.to(device)  
            # model.train()
            loss,_,_,_,_,_,_,output= model(ui, ii,lbl)
            # yc = output.squeeze()
            # loss = criterion(yc, lbl)
            loss_sum += loss.item()
        losses.append(loss_sum / len(ui))
        if(step==0):
            adjusted_loss = loss_sum / len(ui)

        # 恢复原始参数
        for p, original in zip(model.parameters(), original_params):
            p.data = original
            
    for i in range(len(losses)):
        losses[i]=losses[i]-adjusted_loss 

    return losses

def adjust_and_record_loss_RAWP(ii11,model,model_path, ui, ii, lbl,ldr, directions,criterion,testRatings, testNegatives, topK,topK1,topK2,topK3,steps=50, delta=0.01):
    # 加载预训练的参数 
    # # model_path = 'clean_lastfm_1e-05_mim_MLP_robust.pth'
    # print(model_path)
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
        # ui, ii, lbl = iter(ldr).__next__()
        # model.train()
        # loss,_,_,_,_,_,_,output= model(ui, ii,lbl)
        # # yc = output.squeeze()
        # # loss = criterion(yc, lbl)
        # loss_sum += loss.item()
        for ui, ii, lbl in ldr:
            ui, ii, lbl = ui.to(device), ii.to(device), lbl.to(device)  
            # model.train()
            loss,_,_,_,_,_,_,output= model(ui, ii,lbl)
            # yc = output.squeeze()
            # loss = criterion(yc, lbl)
            loss_sum += loss.item()
        losses.append(loss_sum / len(ui))
        if(step==0):
            adjusted_loss = loss_sum / len(ui)

        # 恢复原始参数
        for p, original in zip(model.parameters(), original_params):
            p.data = original
    for i in range(len(losses)):
        losses[i]=losses[i]-adjusted_loss
    return losses


if __name__ == "__main__":
    main()

import argparse
import logging
import sys
import time
import math
from matplotlib import pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.ndimage import gaussian_filter1d
import os
from mlp_test import MLP
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
    parser.add_argument('--batch-size', default=100, type=int)    # 100
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
    parser.add_argument("--num_samples", type=int, default=0,help="Number of negative instances to pair with a positive instance.")
 
    return parser.parse_args()


def main(num_rounds):
    All_start_time = time.time() 
    args = get_args()
    fcLayers = eval(args.fcLayers)
    topK = 10
    topK1 = 20
    topK2 = 50
    topK3 = 100
    # 微调的步长和范围
    steps = 50 # 50
    delta = 0.01
    rounds = 300 # 200
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    
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
    
    
    # model_path1 = 'ml-1m_0.0001_fgsm_MLP_clean_robust_0521_1.pth'
    # model_path2 = 'ml-1m_0.0005_fgsm_L2_robust_0529_1.pth'
    # # model_path7 = '80_ml-1m_MLP_init_1713513910.3174512.pth'
    # model_path8 = 'pretrained/ml-1m_RAWP_FineTurning_test80_2.pth'
    
    model_path1 = "AT_AWP/param_file/RAWP/80_ml-1m_MLP_init_1713513910.3174512.pth"  # RAWP
    model_path2 = "AT_AWP/param_file/RAWP-FT/ml-1m_RAWP_80_FineTurning_1715970106.2888427.pth" # RAWP-FT
    model_path3 = "AT_AWP/param_file/MLP/93_ml-1m_0.0001_fgsm_MLP_clean_robust_0521_1.pth" # MLP
    model_path4 = "AT_AWP/param_file/RAT/ml-1m_RAT_0.0005_fgsm_L2_robust_0529_1.pth" # RAT
    model_path8 = "AT_AWP/param_file/best_model/ml-1m_FTS_lr-0.005_gamma-0.001_Best.pth" # FTS

    # model_path1 = "param_file/RAWP/80_ml-1m_MLP_init_1713513910.3174512.pth"
    # model_path2 = "param_file/RAWP-FT/ml-1m_RAWP_80_FineTurning_1715970106.2888427.pth"
    # model_path8 = "param_file/FTS/ML-1M/ml-1m_FTS_lr-0.005_gamma-0.001.pth"

    # model_paths = [model_path3, model_path4, model_path1, model_path2, model_path8]
    model_paths = [model_path4, model_path1, model_path2, model_path8]  # 不带MLP
    # model_labels = ['MLP', 'RAT', 'RAWP', 'RAWP-FT', 'FTS']
    model_labels = ['MLP+RAT', 'MLP+RAWP', 'MLP+RAWP-FT', 'MLP+RAWP-PDAF']
    losses_avg = np.zeros((len(model_paths), 2*steps+1))
    loss_functions = [
        # adjust_and_record_loss_RAWP,  # MLP
        adjust_and_record_loss,       # RAT
        adjust_and_record_loss_RAWP,  # RAWP
        adjust_and_record_loss_RAWP,  # RAWP-FT
        adjust_and_record_loss_RAWP   # FTS
    ]
    ui, ii, lbl = iter(ldr).__next__()
    ui, ii, lbl = ui.to(device), ii.to(device), lbl.to(device)
    for round_num in range(rounds):
        directions = []
        for p in model_adv1.parameters():
            # 从高斯分布中为p的形状采样
            d = torch.randn_like(p)
            # 对采样得到的方向进行Frobenius范数归一化
            norm = torch.norm(d, p=2)
            d_normalized = d / (norm + 1e-20)  # 添加小的常数防止除以零
            directions.append(d_normalized)     
        
        for i, (model_path, label,loss_function) in enumerate(zip(model_paths, model_labels,loss_functions)):
            losses_model = loss_function(round_num,model_adv1, model_path, ui, ii, lbl,ldr, directions, criterion, testRatings, testNegatives, topK, topK1, topK2, topK3, steps, delta)
            losses_avg[i] += np.array(losses_model)
        
        torch.cuda.empty_cache()
        print(f"rounds:{round_num}, time:{time.time() - t1}")
        
    losses_avg /= rounds
    x_axis = np.linspace(-steps*delta, steps*delta, 2*steps+1)
    losses_avg = np.array(losses_avg)
    losses_avg[0] = gaussian_filter1d(losses_avg[0], sigma=1)# 高斯滤波 2,segam,越大滤波越厉害
    losses_avg[1] = gaussian_filter1d(losses_avg[1], sigma=1)# 高斯滤波 2,segam,越大滤波越厉害
    losses_avg[2] = gaussian_filter1d(losses_avg[2], sigma=1)# 高斯滤波 2,segam,越大滤波越厉害
    losses_avg[3] = gaussian_filter1d(losses_avg[3], sigma=1)# 高斯滤波 2,segam,越大滤波越厉害
    # losses_avg[4] = gaussian_filter1d(losses_avg[4], sigma=2)# 高斯滤波 2,segam,越大滤波越厉害
    # hxnsuihb = losses_avg[0]steps
    for i in range(2*steps+1):
        losses_avg[0][i]=losses_avg[0][i]-losses_avg[0][steps]
    for i in range(2*steps+1):
        losses_avg[1][i]=losses_avg[1][i]-losses_avg[1][steps]
    for i in range(2*steps+1):
        losses_avg[2][i]=losses_avg[2][i]-losses_avg[2][steps]
    for i in range(2*steps+1):
        losses_avg[3][i]=losses_avg[3][i]-losses_avg[3][steps]
    # for i in range(2*steps+1):
    #     losses_avg[4][i]=losses_avg[4][i]-losses_avg[4][steps]
    # 绘制损失Landscape
    for i, label in enumerate(model_labels):
        plt.plot(x_axis, losses_avg[i], label=label)
    plt.ylim(-0.00005, 0.0008)
    plt.xlabel('Weight adjustment in the direction')
    plt.ylabel('Loss')
    plt.title('Loss Landscape of MLP Recommendation Model')
    plt.legend()
    # path_save = f'AT_AWP/output/loss_train_landscape_1D_ml1m_bsz{args.batch_size}seed{args.seed}rounds:{rounds}_{steps}_{delta}_{time.time()}.png'
    path_save = f'AT_AWP/output/landscape_num-{args.num_samples}_{num_rounds}.png'
    plt.savefig(path_save)
    print(path_save)
    plt.clf()
    plt.show()
    

def adjust_and_record_loss(ii11,model,model_path, ui, ii, lbl,ldr, directions,criterion,testRatings, testNegatives, topK,topK1,topK2,topK3,steps=50, delta=0.01):
    original_state_dict = torch.load(model_path, map_location=device)
    
    # 创建一个新的state_dict，只包含不带有'reg'字段的参数
    new_state_dict111 = {}
    for key, value in original_state_dict.items():
        if 'reg' not in key:
            new_state_dict111[key] = value
    # 修改键以匹配 DataParallel 的期望格式
    # print(model_path)
    if model_path == "AT_AWP/param_file/RAWP-FT/ml-1m_RAWP_80_FineTurning_1715970106.2888427.pth" or model_path == "AT_AWP/param_file/MLP/93_ml-1m_0.0001_fgsm_MLP_clean_robust_0521_1.pth" or model_path == "AT_AWP/param_file/RAT/ml-1m_RAT_0.0005_fgsm_L2_robust_0529_1.pth":
        modified_state_dict = {'module.' + key: value for key, value in new_state_dict111.items()}
    else:
        modified_state_dict = new_state_dict111
    # 现在使用修改后的状态字典加载模型
    model.load_state_dict(modified_state_dict) 
    model.to(device)
    model.eval()
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
        loss,_,_,_,_,_,_,output= model(ui, ii,lbl)
        loss_sum += loss.item()
        
        # for ui, ii, lbl in ldr:
        #     ui, ii, lbl = ui.to(device), ii.to(device), lbl.to(device)  
        #     # model.train()
        #     loss,_,_,_,_,_,_,output= model(ui, ii,lbl)
        #     # yc = output.squeeze()
        #     # loss = criterion(yc, lbl)
        #     loss_sum += loss.item()
        
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

    # # ti = time.time()
    # original_state_dict = torch.load(model_path,map_location=device)
    #     # 创建一个新的state_dict，只包含不带有'reg'字段的参数
    # new_state_dict111 = {}
    # for key, value in original_state_dict.items():
    #     if 'reg' not in key:
    #         new_state_dict111[key] = value
    # # model.load_state_dict(new_state_dict111)
    
    # modified_state_dict = {'module.' + key: value for key, value in new_state_dict111.items()}
    # # 现在使用修改后的状态字典加载模型
    # model.load_state_dict(modified_state_dict) 
    # # print(time.time()-ti)
    
    if 'robust' in model_path:  
        # 假设 original_state_dict 是你加载的原始状态字典 
        original_state_dict = torch.load(model_path, map_location=device)
        
        # # 如果是mlp生成的模型，有net需要提取出来
        # original_state_dict = original_state_dict['net']
        
        # 修改键以匹配 DataParallel 的期望格式 # 新修改，不需要这一步操作
        # modified_state_dict = {'module.' + key: value for key, value in original_state_dict.items()}\
        # modified_state_dict = original_state_dict
        # print(model_path)
        if model_path == "AT_AWP/param_file/RAWP-FT/ml-1m_RAWP_80_FineTurning_1715970106.2888427.pth" or model_path == "AT_AWP/param_file/MLP/93_ml-1m_0.0001_fgsm_MLP_clean_robust_0521_1.pth" or model_path == "AT_AWP/param_file/RAT/ml-1m_RAT_0.0005_fgsm_L2_robust_0529_1.pth":
            modified_state_dict = {'module.' + key: value for key, value in original_state_dict.items()}
        else:
            modified_state_dict = original_state_dict
        
        # 创建一个新的state_dict，只包含不带有'reg'字段的参数
        new_state_dict111 = {}
        for key, value in modified_state_dict.items():
            if 'reg' not in key:
                new_state_dict111[key] = value
        model.load_state_dict(new_state_dict111)
        
        # # 现在使用修改后的状态字典加载模型
        # model_adv.load_state_dict(modified_state_dict) 
    else:  # RAWP
        original_state_dict = torch.load(model_path,map_location=device)
        # 创建一个新的state_dict，只包含不带有'reg'字段的参数
        new_state_dict111 = {}
        for key, value in original_state_dict.items():
            if 'reg' not in key:
                new_state_dict111[key] = value
        model.load_state_dict(new_state_dict111)
    
    model.to(device)
    model.eval()

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
        loss,_,_,_,_,_,_,output= model(ui, ii,lbl)
        # yc = output.squeeze()
        # loss = criterion(yc, lbl)
        loss_sum += loss.item()
        
        # for ui, ii, lbl in ldr:
        #     ui, ii, lbl = ui.to(device), ii.to(device), lbl.to(device)  
        #     # model.train()
        #     loss,_,_,_,_,_,_,output= model(ui, ii,lbl)
        #     # yc = output.squeeze()
        #     # loss = criterion(yc, lbl)
        #     loss_sum += loss.item()
        losses.append(loss_sum / len(ui))
        if(step==0):
            adjusted_loss = loss_sum / len(ui)

        # 恢复原始参数
        for p, original in zip(model.parameters(), original_params):
            p.data = original
    for i in range(len(losses)):
        losses[i]=losses[i]-adjusted_loss
    # loss_differences_1 = np.array(losses)
    # loss_differences_1 = gaussian_filter1d(loss_differences_1, sigma=2)# 高斯滤波 2,segam,越大滤波越厉害
    # return loss_differences_1
    return losses

if __name__ == "__main__":
    for i in range(10):
        main(i)

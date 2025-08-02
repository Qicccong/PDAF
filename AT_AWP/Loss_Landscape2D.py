import argparse
import logging
import pickle
import sys
import time

from bunch import Bunch
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os
from mlp import MLP
from Dataset import Dataset

from preactresnet import PreActResNet18
from evaluate import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

upper_limit, lower_limit = 1,0


# 定义PGD攻击函数
def pgd_attack(model, x, y_true, epsilon, alpha, num_steps):
    delta = torch.zeros_like(x, requires_grad=True)
    for _ in range(num_steps):
        loss = nn.CrossEntropyLoss()(model(x + delta), y_true)
        loss.backward()
        delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.grad.zero_()
    return delta.detach()


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

class Batches():
    def __init__(self, dataset, batch_size, shuffle, set_random_choices=False, num_workers=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle, drop_last=drop_last
        )

    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices()
        return ({'input': x.to(device).float(), 'target': y.to(device).long()} for (x,y) in self.dataloader)

    def __len__(self):
        return len(self.dataloader)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='MLP')
    parser.add_argument('--l2', default=0, type=float)
    parser.add_argument('--l1', default=0, type=float)
    parser.add_argument('--batch-size', default=1024, type=int)
    parser.add_argument("--data_path", nargs="?", default="Data/",
                        help="Input data path.")
    parser.add_argument("--dataset", nargs="?", default="ml-1m",   # ml-1m yelp  AToy lastfm
                        help="Choose a dataset.")
    # parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr-max', default=0.005, type=float)        # 0.1 这块应该是推荐模型训练学习率 RAT ml1m 0.0005
    parser.add_argument('--lr-prox', default=0.01, type=float)
    parser.add_argument('--attack', default='fgsm', type=str, choices=['pgd', 'fgsm', 'free', 'none'])
    parser.add_argument('--epsilon', default=0.5, type=int)
    parser.add_argument("--alpha", nargs="?", default=0.5/25)  # 0.5/25
    parser.add_argument("--num_steps", nargs="?", default=25)  #25
    # parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'])
    parser.add_argument('--fname', default='mlp_model', type=str)
    parser.add_argument('--seed', default=3, type=int)
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
    steps = 20
    delta = 0.01

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
        model_adv2 = MLP(fcLayers, userMatrix, itemMatrix,device)
    else:
        raise ValueError("Unknown model")

    model_adv1 = nn.DataParallel(model_adv1).to(device)
    model_adv2 = nn.DataParallel(model_adv2).to(device)

    criterion = torch.nn.BCELoss()

    # 加载预训练的参数 
    # model_path1 = '/mnt/home/E22301274/Def/AWP-main/64_lastfm_MLP_clean_8840_1710387934.3992233.pth'
    # model_path1 = '63_rawp_lastfm_MLP_init_1710319680.1585555.pth'
    model_path1 = "AT_AWP\param_file\RAWP\80_ml-1m_MLP_init_1713513910.3174512.pth"  #RAWP
    print(model_path1)
    # model_path2 = 'lastfm_MLP_init_1710311742.536004.pth'  # 62
    model_path2 = "AT_AWP\param_file\RAWP-FT\ml-1m_RAWP_80_FineTurning_1715970106.2888427.pth"
    print(model_path2)
    # model_path3 = 'lastfm_MLP_adv_1709210611.3623106.pth'
    # print(model_path3)
    # model_path3 = 'lastfm_1e-05_mim_MLP_robust_eval.pth'
    model_path3 = "AT_AWP\param_file\FTS\ML-1M\ml-1m_FTS_lr-0.005_gamma-0.001.pth"
    print(model_path3)
    model_names = ['RWAP8845','RWAP7020','RAT']

    # 选择一个方向
    direction1 = [torch.randn_like(p) for p in model_adv1.parameters()]
    direction2 = [torch.randn_like(p) for p in model_adv1.parameters()]
    
    userInput, itemInput, labels = get_train_instances(train, args.nNeg)
    dst = BatchDataset(userInput, itemInput, labels)
    ldr = torch.utils.data.DataLoader(dst, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    model_paths = [model_path1, model_path2, model_path3]
    loss_functions = [adjust_and_record_loss_RAWP, adjust_and_record_loss_RAWP, adjust_and_record_loss]

    for i, (model_name,model_path, loss_function) in enumerate(zip(model_names,model_paths, loss_functions), start=1):
        losses_model = loss_function(model_adv1, model_path, ldr, direction1, direction2, criterion, testRatings, testNegatives, topK, topK1, topK2, topK3, steps, delta)

        x = np.linspace(-steps * delta, steps * delta, 2 * steps + 1)
        y = np.linspace(-steps * delta, steps * delta, 2 * steps + 1)
        X, Y = np.meshgrid(x, y)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, losses_model, cmap='viridis')
        ax.set_xlabel('Direction 1')
        ax.set_ylabel('Direction 2')
        ax.set_zlabel('Loss')
        plt.title(f'3D Loss Landscape_{model_name}')
        path_save = f'loss_landscape_lastfm_{model_name}_{steps}_{delta}_{time.time()}.png'
        plt.savefig(path_save)
        print(path_save)
        print(time.time()-All_start_time)
        # 打开一个文件以写入内容，如果文件不存在将被创建
        with open(f'_{args.seed}_{model_name}_{steps}_{delta}_log.txt', 'w') as f:
            # 遍历二维列表中的每一行
            for row in losses_model:
                # 将每一行的元素转换为字符串，并用逗号分隔
                row_str = ', '.join(str(item) for item in row)
                # 写入转换后的字符串到文件，并添加换行符以分隔行
                f.write(row_str + '\n')

        print(f'二维列表已保存到 _{args.seed}_{model_name}_{steps}_{delta}_log.txt.txt 文件中。')
    
def adjust_and_record_loss(model,model_path, ldr, direction1,direction2,criterion,testRatings, testNegatives, topK,topK1,topK2,topK3,steps=50, delta=0.01):
    # 加载预训练的参数 
    # model_path = 'clean_lastfm_1e-05_mim_MLP_robust.pth'
    print(model_path)
    # 假设 original_state_dict 是你加载的原始状态字典
    original_state_dict = torch.load(model_path)
    # 修改键以匹配 DataParallel 的期望格式
    modified_state_dict = {'module.' + key: value for key, value in original_state_dict.items()}
    # 现在使用修改后的状态字典加载模型
    model.load_state_dict(modified_state_dict) 
    
    # Check  performance
    t1 = time.time()
    hits10, ndcgs10, maps10, mrrs10, hits20, ndcgs20,maps20, mrrs20,hits50, ndcgs50,maps50, mrrs50 = evaluate_model(model, testRatings, testNegatives, topK,topK1,topK2,topK3, num_thread=1, device=device)
    hr10, ndcg10,map10,mrr10, hr20, ndcg20,map20,mrr20,hr50, ndcg50,map50, mrr50 = np.array(hits10).mean(), np.array(ndcgs10).mean(),np.array(maps10).mean(), np.array(mrrs10).mean(),\
        np.array(hits20).mean(), np.array(ndcgs20).mean(), np.array(maps20).mean(), np.array(mrrs20).mean(), np.array(hits50).mean(), np.array(ndcgs50).mean(), np.array(maps50).mean(), np.array(mrrs50).mean()
    print(f"model Init: HR10={hr10:.4f}, NDCG10={ndcg10:.4f}, mrrs10={mrr10:.4f}, HR20={hr20:.4f}, NDCG20={ndcg20:.4f}, mrrs20={mrr20:.4f}, HR50={hr50:.4f}, NDCG50={ndcg50:.4f}, mrrs50={mrr50:.4f} [{time.time()-t1:.1f}s]")
    
    
    # 初始化一个二维数组来存储损失值
    losses = np.zeros((2*steps+1, 2*steps+1))
    
    for i, step1 in enumerate(range(-steps, steps+1)):
        for j, step2 in enumerate(range(-steps, steps+1)):
            # 保存原始参数
            original_params = [p.data.clone() for p in model.parameters()]
            
            # 在两个方向上微调权重
            for p, d1, d2 in zip(model.parameters(), direction1, direction2):
                p.data += d1 * delta * step1 + d2 * delta * step2

            # 计算损失
            loss_sum = 0
            for ui, ii, lbl in ldr:
                ui, ii, lbl = ui.to(device), ii.to(device), lbl.to(device)  
                loss,_,_,_,_,_,_,output= model(ui, ii,lbl)
                # yc = output.squeeze()
                # loss = criterion(yc, lbl)
                loss_sum += loss.item()
            losses[i, j] = loss_sum / len(ldr)
            # 恢复原始参数
            for p, original in zip(model.parameters(), original_params):
                p.data = original
    return losses

def adjust_and_record_loss_RAWP(model,model_path, ldr, direction1,direction2,criterion,testRatings, testNegatives, topK,topK1,topK2,topK3,steps=50, delta=0.01):
    # 加载预训练的参数 
    model.load_state_dict(torch.load(model_path))  
    
    # Check  performance
    t1 = time.time()
    hits10, ndcgs10, maps10, mrrs10, hits20, ndcgs20,maps20, mrrs20,hits50, ndcgs50,maps50, mrrs50 = evaluate_model(model, testRatings, testNegatives, topK,topK1,topK2,topK3, num_thread=1, device=device)
    hr10, ndcg10,map10,mrr10, hr20, ndcg20,map20,mrr20,hr50, ndcg50,map50, mrr50 = np.array(hits10).mean(), np.array(ndcgs10).mean(),np.array(maps10).mean(), np.array(mrrs10).mean(),\
        np.array(hits20).mean(), np.array(ndcgs20).mean(), np.array(maps20).mean(), np.array(mrrs20).mean(), np.array(hits50).mean(), np.array(ndcgs50).mean(), np.array(maps50).mean(), np.array(mrrs50).mean()
    print(f"model Init: HR10={hr10:.4f}, NDCG10={ndcg10:.4f}, mrrs10={mrr10:.4f}, HR20={hr20:.4f}, NDCG20={ndcg20:.4f}, mrrs20={mrr20:.4f}, HR50={hr50:.4f}, NDCG50={ndcg50:.4f}, mrrs50={mrr50:.4f} [{time.time()-t1:.1f}s]")
    
    # 初始化一个二维数组来存储损失值
    losses = np.zeros((2*steps+1, 2*steps+1))
    
    for i, step1 in enumerate(range(-steps, steps+1)):
        for j, step2 in enumerate(range(-steps, steps+1)):
            # 保存原始参数
            original_params = [p.data.clone() for p in model.parameters()]
            
            # 在两个方向上微调权重
            for p, d1, d2 in zip(model.parameters(), direction1, direction2):
                p.data += d1 * delta * step1 + d2 * delta * step2

            # 计算损失
            loss_sum = 0
            for ui, ii, lbl in ldr:
                ui, ii, lbl = ui.to(device), ii.to(device), lbl.to(device)  
                loss,_,_,_,_,_,_,output= model(ui, ii,lbl)
                # yc = output.squeeze()
                # loss = criterion(yc, lbl)
                loss_sum += loss.item()
            losses[i, j] = loss_sum / len(ldr)

            # 恢复原始参数
            for p, original in zip(model.parameters(), original_params):
                p.data = original

    return losses



if __name__ == "__main__":
    main()

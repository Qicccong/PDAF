from copy import deepcopy
import copy
import heapq
import logging
import math
import multiprocessing
import time
import numpy as np
from collections import namedtuple
import torch
# from torch import nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
# import pandas as pd
# from scipy import sparse
import numpy as np


def cal_lp_norm(tensor,p,dim_count):
    tmp = tensor
    for i in range(1,dim_count):
        tmp = torch.norm(tmp,p=p,dim=i,keepdim=True) #torch.Size([100, 1])
    
    
    return torch.clamp_min(tmp, 1e-8)

# evaluate

# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_K = None
_K1 = None
_K2 = None
_K3 = None

def evaluate_model(model, testRatings, testNegatives, K, K1, K2,K3,attack1,epsilon1,alpha1,num_steps1,decay_factor, num_thread, device):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _testNegatives
    global _K
    global _K1
    global _K2
    global _K3
    global model_stabel
    global attack
    global epsilon
    global alpha
    global num_steps
    global _model
    global _decay_factor
    model_stabel = model
    # _model = copy.deepcopy(model)
    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K
    _K1 = K1
    _K2 = K2
    _K3 = K3
    attack =  attack1
    epsilon = epsilon1
    alpha =  alpha1
    num_steps = num_steps1
    _decay_factor =decay_factor
    
    # np.random.seed(123)
    # torch.manual_seed(123)
    # torch.cuda.manual_seed(123)
    
    hits10, ndcgs10,maps10, mrrs10, hits20, ndcgs20,maps20, mrrs20, hits50, ndcgs50, maps50, mrrs50 = [], [], [], [], [], [], [], [],[], [], [], []
    if num_thread > 1:  # Multi-thread
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating, range(len(_testRatings)))
        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        return (hits, ndcgs)
    # Single thread
    for idx in range(len(_testRatings)):
        # (hr10, ndcg10, ap10, mrr10, hr20, ndcg20,ap20, mrr20, hr50, ndcg50, ap50, mrr50) = eval_one_rating(idx, device)
        (hr10, ndcg10, ap10, mrr10) = eval_one_rating(idx, device)
        hits10.append(hr10)
        ndcgs10.append(ndcg10)
        maps10.append(ap10)
        mrrs10.append(mrr10)

        # hits20.append(hr20)
        # ndcgs20.append(ndcg20)
        # maps20.append(ap20)
        # mrrs20.append(mrr20)

        # hits50.append(hr50)
        # ndcgs50.append(ndcg50) 
        # maps50.append(ap50)
        # mrrs50.append(mrr50)       
    # return (hits10, ndcgs10, maps10, mrrs10, hits20, ndcgs20,maps20, mrrs20,hits50, ndcgs50,maps50, mrrs50)
    return (hits10, ndcgs10, maps10, mrrs10)

def broadcast_multiply(tensor_a, tensor_b):
    # 确保在相同设备
    tensor_b = tensor_b.to(tensor_a.device)
    
    # 扩展维度
    if tensor_b.dim() < tensor_a.dim():
        # 在末尾添加维度直到匹配
        expand_dims = [1] * (tensor_a.dim() - tensor_b.dim())
        tensor_b = tensor_b.view(*tensor_b.shape, *expand_dims)
    
    return tensor_a * tensor_b

def eval_one_rating(idx, device):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    # u = rating[0]
    # gtItem = rating[1]
    r = rating[2]
    # items.append(gtItem)
    # # Get prediction scores
    # map_item_score = {}
    # users = np.full(len(items), u, dtype='int32')
    # label = np.full(len(items), r, dtype='int32')
    # # ---
    # dst = TestDataset_adv(users, items, label)
    # ldr = torch.utils.data.DataLoader(dst, batch_size=256, shuffle=False)
    
    # labels = list(0 for _ in range(len(_testNegatives[idx])))
    u = rating[0]
    gtItem = rating[1]
    items.append(gtItem)
    # gtlabel = rating[2]
    # labels.append(gtlabel)
    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items), u, dtype='int32')
    # label = np.full(len(items), r, dtype='int32')
    lbl = np.full(len(items), 0, dtype='float32')
    with torch.no_grad():
        # _,_,_,_,_,_, ly = _model.model(users,items)
        users1 = torch.tensor(users, dtype=torch.long).to(device)
        items1 = torch.tensor(items, dtype=torch.long).to(device)
     
        lbl = torch.tensor(lbl)
        loss,*_,ly= model_stabel(users1, items1,lbl)
    # labels = 
    # labels = ly.tolist()
    ly_squeezed = ly.squeeze()
    labels1 = ly_squeezed.tolist()
    # labels = ly.numpy()
    # labels = torch.unbind(ly)
    # labels[99]=1
    labels = [1 if value > 0.5 else 0 for value in labels1]
    
    isCuda = torch.cuda.is_available()
    dst = TestDataset_adv(users, items,labels)
    ldr = torch.utils.data.DataLoader(dst, batch_size=100, shuffle=False)
    _model = copy.deepcopy(model_stabel)
    # _model = model_stabel
    # _model.eval()
    predictions = [None] * len(dst)
    total = 0
    
    # _model要经过一次反向传播更新模型参数再到下面进行预测评测
    # attack = 'bim'
    # epsilon = 0.1 # args.epsilon# .4  # 扰动的最大范围
    # alpha = 0.02# args.alpha # 1# 0.02   # 每步更新的步长  epsilon/num_steps
    # num_steps = 5 # args.num_steps # 1 #25 # 迭代次数  RAT是25轮
    # print(f"epsilon: {epsilon:.4f} , alpha: {alpha:.4f} num_steps:{num_steps:.4f} ")
    # emb_backup = {} 
    # for name, param in _model.named_parameters():
    #     if "weight" in name:
    #             emb_backup[name] = param.data.clone()
    
    
    if attack == 'fgsm': 
        emb_backup = {} 
        is_first_attack = True
        # 对模型参数进行 fgsm 攻击
        start_time = time.time()
        # Generate training instances
        for ui, ii, lbl in ldr:
            ui, ii, lbl = ui.to(device), ii.to(device), lbl.to(device)  
            _model.train()
            loss,*_,output= _model(ui, ii,lbl)
            loss.backward()
            with torch.no_grad():
                # for param in model_adv.parameters():
                for name, param in _model.named_parameters():
                    if "weight" in name:
                        if is_first_attack:
                            emb_backup[name] = param.data.clone()
                        norm = torch.norm(param.grad)
                        if norm != 0 and not torch.isnan(norm):
                            # l2方式固定噪声大小
                            # size_para = param.size(0)
                            # r_at = (epsilon) * (param.grad / cal_lp_norm(param.grad,p=2,dim_count=len(param.size())))
                            # normVal = torch.norm(r_at.view(size_para, -1), 2, 1)
                            # mask = normVal<=epsilon
                            # scaling = epsilon/normVal
                            # scaling[mask] = 1
                            # r_at = r_at*scaling.view(size_para, 1)
                            # param.data = emb_backup[name] + r_at
                            
                            # # 单linf方式固定噪声大小
                            # r_at = param.grad.sign()*epsilon
                            # r_at = torch.clamp(r_at, -epsilon, epsilon).to(device) 
                            # param.data = emb_backup[name] + r_at
                            
                    
                            # 按照比例设置固定噪声大小linf
                            size_para = param.size(0)
                            # r_at = param.grad.sign()
                            r_at = (param.grad / cal_lp_norm(param.grad,p=2,dim_count=len(param.size())))
                            normVal1 = torch.norm(r_at.view(size_para, -1), 2, 1)
                            normVal2 = torch.norm(param.view(size_para, -1), 2, 1)
                            # mask = normVal1<=normVal2 * self.epsilon
                            scaling = normVal2/normVal1 * epsilon   # 算出一次噪声需要调整的倍数
                            scaling[scaling == float('inf')] = 0
                            # scaling[mask] = 1

                           

                            # print(f"r_at shape: {r_at.shape}")
                            # print(f"scaling shape: {scaling.shape}")
                            # print(f"size_para: {size_para}")
                            # r_at = r_at*scaling.view(size_para, 1)
                            r_at = broadcast_multiply(r_at, scaling)
                            param.data = emb_backup[name] + r_at # 把这一次的噪声加到参数上
                            
                            # delta = param.data - emb_backup[name]
                            # normVal1 = torch.norm(delta.view(size_para, -1), 2, 1)
                            # normVal2 = torch.norm(param.view(size_para, -1), 2, 1)
                            # # mask = normVal1<=normVal2 * self.epsilon
                            # scaling = normVal2/normVal1 * epsilon   
                            # scaling[scaling == float('inf')] = 0
                            # # scaling[mask] = 1
                            # delta = delta*scaling.view(size_para, 1)
                            # param.data = emb_backup[name] + delta
                            
                        
                is_first_attack = False
        # print(time.time()-start_time)
    
    
    if attack == 'bim':
        emb_backup = {} 
        is_first_attack = True
        # 对模型参数进行 PGD 攻击
        for epoch in range(num_steps):
            start_time = time.time()
            # Generate training instances
            for ui, ii, lbl in ldr:
                ui, ii, lbl = ui.to(device), ii.to(device), lbl.to(device)  
                _model.train()
                loss,*_,output= _model(ui, ii,lbl)
                yc = output.squeeze()
                loss.backward()
                # opt.step()
                # 对模型的每个参数进行扰动            
                with torch.no_grad():
                    # for param in model_adv.parameters():
                    for name, param in _model.named_parameters():
                        if "weight" in name:
                            if is_first_attack:
                                emb_backup[name] = param.data.clone()
                            norm = torch.norm(param.grad)
                            if norm != 0 and not torch.isnan(norm):
                                # size_para = param.size(0)
                                # param.data = param.data + (epsilon/num_steps) * (param.grad / cal_lp_norm(param.grad,p=2,dim_count=len(param.size())))
                                # r_at = param.data - emb_backup[name]
                                # normVal = torch.norm(r_at.view(size_para, -1), 2, 1)
                                # mask = normVal<=epsilon
                                # scaling = epsilon/normVal
                                # scaling[mask] = 1
                                # r_at = r_at*scaling.view(size_para, 1)
                                # param.data = emb_backup[name] + r_at
                                
                                # # 单linf方式固定噪声大小
                                # param.data = param.data + param.grad.sign()*(epsilon/num_steps)
                                # # r_at = param.grad.sign()*(epsilon/num_steps)
                                # r_at = param.data - emb_backup[name]
                                # r_at = torch.clamp(r_at, -epsilon, epsilon).to(device) 
                                # param.data = emb_backup[name] + r_at
                                
                                # 按照比例设置固定噪声大小linf
                                size_para = param.size(0)
                                # r_at = param.grad.sign()
                                r_at = (param.grad / cal_lp_norm(param.grad,p=2,dim_count=len(param.size())))
                                normVal1 = torch.norm(r_at.view(size_para, -1), 2, 1)
                                normVal2 = torch.norm(param.view(size_para, -1), 2, 1)
                                # mask = normVal1<=normVal2 * self.epsilon
                                scaling = normVal2/normVal1 * epsilon/num_steps   # 算出一次噪声需要调整的倍数
                                scaling[scaling == float('inf')] = 0
                                # scaling[mask] = 1
                                r_at = r_at*scaling.view(size_para, 1)
                                param.data = param.data + r_at # 把这一次的噪声加到参数上
                                
                                delta = param.data - emb_backup[name]
                                normVal1 = torch.norm(delta.view(size_para, -1), 2, 1)
                                normVal2 = torch.norm(param.view(size_para, -1), 2, 1)
                                mask = normVal1<=normVal2 * epsilon
                                scaling = normVal2/normVal1 * epsilon
                                scaling[scaling == float('inf')] = 0
                                scaling[mask] = 1  # 
                                delta = delta*scaling.view(size_para, 1)
                                
                                normVal1 = torch.norm(delta.view(size_para, -1), 2, 1)
                                param.data = emb_backup[name] + delta
                                
                    is_first_attack = False
    
    if attack == 'pgd':
        emb_backup = {}
        is_first_attack = True
        random_start = True
        for epoch in range(num_steps):
            start_time = time.time()
            if is_first_attack:
                # 备份原始模型参数
                with torch.no_grad():
                    for name, param in _model.named_parameters():
                        emb_backup[name] = param.data.clone()
                        if "weight" in name:
                            if random_start and param.requires_grad:
                                # 初始化随机扰动
                                size_para = param.size(0)
                                random_noise = epsilon * torch.randn_like(param)
                                # param.data.add_(random_noise)
                                # # 确保扰动不超过epsilon
                                # r_at = param.data - emb_backup[name]
                                # normVal = torch.norm(r_at.view(size_para, -1), 2, 1)
                                # mask = normVal<=epsilon
                                # scaling = epsilon/normVal
                                # scaling[mask] = 1
                                # r_at = r_at*scaling.view(size_para, 1)
                                # param.data = emb_backup[name] + r_at
                                
                                size_para = param.size(0)
                                r_at = random_noise
                                normVal1 = torch.norm(r_at.view(size_para, -1), 2, 1)
                                normVal2 = torch.norm(param.view(size_para, -1), 2, 1)
                                # mask = normVal1<=normVal2 * self.epsilon
                                scaling = normVal2/normVal1 * epsilon   # 算出一次噪声需要调整的倍数
                                scaling[scaling == float('inf')] = 0
                                # scaling[mask] = 1
                                r_at = r_at*scaling.view(size_para, 1)
                                param.data = emb_backup[name] + r_at # 把这一次的噪声加到参数上
                        
                    # if torch.norm(param_diff) > epsilon:
                    #     param.data = emb_backup[name] + epsilon * param_diff / torch.norm(param_diff)
            
            is_first_attack = False

            for ui, ii, lbl in ldr:
                # ui, ii, lbl = ui.to(device), ii.to(device), lbl.to(device)  
                # _model.train()
                # loss,_,_,_,_,_,_,output= _model(ui, ii,lbl)
                ui, ii, lbl = ui.to(device), ii.to(device), lbl.to(device)  
                _model.train()
                loss,*_,output= _model(ui, ii,lbl)
                yc = output.squeeze()
                loss.backward()
                # loss.backward()
                # 对模型参数应用扰动
                with torch.no_grad():
                    for name, param in _model.named_parameters():
                        if "weight" in name:
                            if param.requires_grad:
                                # 计算扰动
                                norm = torch.norm(param.grad)
                                if norm != 0 and not torch.isnan(norm):
                                    # size_para = param.size(0)
                                    # param.data = param.data + (epsilon/num_steps) * (param.grad / cal_lp_norm(param.grad,p=2,dim_count=len(param.size())))
                                    # r_at = param.data - emb_backup[name]
                                    # normVal = torch.norm(r_at.view(size_para, -1), 2, 1)
                                    # mask = normVal<=epsilon
                                    # scaling = epsilon/normVal
                                    # scaling[mask] = 1
                                    # r_at = r_at*scaling.view(size_para, 1)
                                    # param.data = emb_backup[name] + r_at
                                    # 按照比例设置固定噪声大小linf
                                    size_para = param.size(0)
                                    # r_at = param.grad.sign()
                                    r_at = (param.grad / cal_lp_norm(param.grad,p=2,dim_count=len(param.size())))
                                    normVal1 = torch.norm(r_at.view(size_para, -1), 2, 1)
                                    normVal2 = torch.norm(param.view(size_para, -1), 2, 1)
                                    # mask = normVal1<=normVal2 * self.epsilon
                                    scaling = normVal2/normVal1 * epsilon/num_steps   # 算出一次噪声需要调整的倍数
                                    scaling[scaling == float('inf')] = 0
                                    # scaling[mask] = 1
                                    r_at = r_at*scaling.view(size_para, 1)
                                    param.data = param.data + r_at # 把这一次的噪声加到参数上
                                    
                                    delta = param.data - emb_backup[name]
                                    normVal1 = torch.norm(delta.view(size_para, -1), 2, 1)
                                    normVal2 = torch.norm(param.view(size_para, -1), 2, 1)
                                    mask = normVal1<=normVal2 * epsilon
                                    scaling = normVal2/normVal1 * epsilon
                                    scaling[scaling == float('inf')] = 0
                                    scaling[mask] = 1  # 
                                    delta = delta*scaling.view(size_para, 1)
                                    
                                    normVal1 = torch.norm(delta.view(size_para, -1), 2, 1)
                                    param.data = emb_backup[name] + delta
    
    
    if attack == 'mim':
        emb_backup = {} 
        momentum ={}
        is_first_attack = True
        
        
        for name, param in _model.named_parameters():
                        momentum[name] = torch.zeros_like(param).detach()
        # 对模型参数进行 PGD 攻击
        for epoch in range(num_steps):
            start_time = time.time()
            # Generate training instances
            for ui, ii, lbl in ldr:
                ui, ii, lbl = ui.to(device), ii.to(device), lbl.to(device)  
                _model.train()
                loss,*_,output= _model(ui, ii,lbl)
                yc = output.squeeze()
                loss.backward()
                # opt.step()
                # 对模型的每个参数进行扰动            
                with torch.no_grad():
                    # for param in model_adv.parameters():
                    for name, param in _model.named_parameters():
                        if "weight" in name:
                            if is_first_attack:
                                emb_backup[name] = param.data.clone()
                            norm = torch.norm(param.grad)
                            if norm != 0 and not torch.isnan(norm):
                                
                                param.grad = param.grad + _decay_factor * momentum[name]
                                momentum[name] = param.grad
                                # 按照比例设置固定噪声大小linf
                                size_para = param.size(0)
                                # r_at = param.grad.sign()
                                r_at = (param.grad / cal_lp_norm(param.grad,p=2,dim_count=len(param.size())))
                                normVal1 = torch.norm(r_at.view(size_para, -1), 2, 1)
                                normVal2 = torch.norm(param.view(size_para, -1), 2, 1)
                                # mask = normVal1<=normVal2 * self.epsilon
                                scaling = normVal2/normVal1 * epsilon/num_steps   # 算出一次噪声需要调整的倍数
                                scaling[scaling == float('inf')] = 0
                                # scaling[mask] = 1
                                r_at = r_at*scaling.view(size_para, 1)
                                param.data = param.data + r_at # 把这一次的噪声加到参数上
                                
                                delta = param.data - emb_backup[name]
                                normVal1 = torch.norm(delta.view(size_para, -1), 2, 1)
                                normVal2 = torch.norm(param.view(size_para, -1), 2, 1)
                                mask = normVal1<=normVal2 * epsilon
                                scaling = normVal2/normVal1 * epsilon
                                scaling[scaling == float('inf')] = 0
                                scaling[mask] = 1  # 
                                delta = delta*scaling.view(size_para, 1)
                                
                                normVal1 = torch.norm(delta.view(size_para, -1), 2, 1)
                                param.data = emb_backup[name] + delta
                                
                    is_first_attack = False
    
    # _model.eval()
    with torch.no_grad():
        for ui, ii, lbl in ldr:
            ui, ii, lbl = ui.to(device), ii.to(device), lbl.to(device)
            bsz = ui.size(0)
            if torch.any(lbl > 1):
                lbl[(lbl > 1) | (lbl < 0)] = 0
            *_, ri = _model(ui, ii, lbl)
            # _,ri = _model(ui, ii)
            #ri = criterion(output,lbl)  
            ri = ri.squeeze().cpu().tolist()
            predictions[total:total+bsz] = ri
    # predictions = _model.predict([users, np.array(items)], 
    #                              batch_size=100, verbose=0)
    # ---
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    items.pop()
    
    # Evaluate top rank list
    ranklist10 = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    # ranklist20 = heapq.nlargest(_K1, map_item_score, key=map_item_score.get)
    # ranklist50 = heapq.nlargest(_K2, map_item_score, key=map_item_score.get)

    hr10 = getHitRatio(ranklist10, gtItem)
    ndcg10 = getNDCG(ranklist10, gtItem)
    ap10 = getAP(ranklist10, gtItem)
    mrr10 = getMRR(ranklist10, gtItem)

    # hr20 = getHitRatio(ranklist20, gtItem)
    # ndcg20 = getNDCG(ranklist20, gtItem)
    # ap20 = getAP(ranklist20, gtItem)
    # mrr20 = getMRR(ranklist20, gtItem)

    # hr50 = getHitRatio(ranklist50, gtItem)
    # ndcg50 = getNDCG(ranklist50, gtItem)
    # ap50 = getAP(ranklist50, gtItem)
    # mrr50 = getMRR(ranklist50, gtItem)
 
    # return (hr10, ndcg10, ap10, mrr10, hr20, ndcg20,ap20, mrr20, hr50, ndcg50, ap50, mrr50)
    return (hr10, ndcg10, ap10, mrr10)


def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0


def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0

def getMRR(ranklist, gtItem):
    for index, item in enumerate(ranklist):
        if item == gtItem:
            return 1.0 / (index + 1.0)
    return 0

def getAP(ranklist, gtItem):
    hits = 0
    sum_precs = 0
    for n in range(len(ranklist)):
        if ranklist[n] == gtItem:
            hits += 1
            sum_precs += hits / (n + 1.0)
    if hits > 0:
        return sum_precs / 1
    else:
        return 0



################################################################
## Components from https://github.com/davidcpage/cifar10-fast ##
################################################################

#####################
## data preprocessing
#####################

cifar10_mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616) # equals np.std(train_set.train_data, axis=(0,1,2))/255

def normalise(x, mean=cifar10_mean, std=cifar10_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean*255
    x *= 1.0/(255*std)
    return x

def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border), (0, 0)], mode='reflect')

def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target]) 


def get_train_instances(train, nNeg):
    userInput, itemInput, labels = [], [], []
    nUsers, nItems = train.shape
    for (u, i) in train.keys():
        # positive instance
        userInput.append(u)
        itemInput.append(i)
        labels.append(1)
        # negative instances
        for t in range(nNeg):
            j = np.random.randint(nItems)
            while (u, j) in train.keys():
                j = np.random.randint(nItems)
            userInput.append(u)
            itemInput.append(j)
            labels.append(0)
    return userInput, itemInput, labels

def get_train_matrix(train):
    nUsers, nItems = train.shape
    trainMatrix = np.zeros([nUsers, nItems], dtype=np.int32)
    for (u, i) in train.keys():
        trainMatrix[u][i] = 1
    return trainMatrix

#####################
## data augmentation
#####################



class Crop(namedtuple('Crop', ('h', 'w'))):
    def __call__(self, x, x0, y0):
        return x[:,y0:y0+self.h,x0:x0+self.w]

    def options(self, x_shape):
        C, H, W = x_shape
        return {'x0': range(W+1-self.w), 'y0': range(H+1-self.h)}
    
    def output_shape(self, x_shape):
        C, H, W = x_shape
        return (C, self.h, self.w)
    
class FlipLR(namedtuple('FlipLR', ())):
    def __call__(self, x, choice):
        return x[:, :, ::-1].copy() if choice else x 
        
    def options(self, x_shape):
        return {'choice': [True, False]}

class Cutout(namedtuple('Cutout', ('h', 'w'))):
    def __call__(self, x, x0, y0):
        x = x.copy()
        x[:,y0:y0+self.h,x0:x0+self.w].fill(0.0)
        return x

    def options(self, x_shape):
        C, H, W = x_shape
        return {'x0': range(W+1-self.w), 'y0': range(H+1-self.h)} 
    
    
class Transform():
    def __init__(self, dataset, transforms):
        self.dataset, self.transforms = dataset, transforms
        self.choices = None
        
    def __len__(self):
        return len(self.dataset)
           
    def __getitem__(self, index):
        data, labels = self.dataset[index]
        for choices, f in zip(self.choices, self.transforms):
            args = {k: v[index] for (k,v) in choices.items()}
            data = f(data, **args)
        return data, labels
    
    def set_random_choices(self):
        self.choices = []
        x_shape = self.dataset[0][0].shape
        N = len(self)
        for t in self.transforms:
            options = t.options(x_shape)
            x_shape = t.output_shape(x_shape) if hasattr(t, 'output_shape') else x_shape
            self.choices.append({k:np.random.choice(v, size=N) for (k,v) in options.items()})

#####################
## dataset
#####################

def cifar10(root):
    train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
    return {
        'train': {'data': train_set.data, 'labels': train_set.targets},
        'test': {'data': test_set.data, 'labels': test_set.targets}
    }

#####################
## data loading
#####################

class BatchDataset(torch.utils.data.Dataset):
    def __init__(self, userInput, itemInput, labels):
        super(BatchDataset, self).__init__()
        self.userInput = torch.Tensor(userInput).long()
        self.itemInput = torch.Tensor(itemInput).long()
        self.labels = torch.Tensor(labels)

    def __getitem__(self, index):
        return self.userInput[index], self.itemInput[index], self.labels[index]

    def __len__(self):
        return self.labels.size(0)


class AverageMeter(object):
    """Adapted from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self, name, fmt=".4f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = f"{self.name} {self.val:{self.fmt}} ({self.avg:{self.fmt}})"
        return fmtstr


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
        return ({'input': x.to(device).half(), 'target': y.to(device).long()} for (x,y) in self.dataloader)
    
    def __len__(self): 
        return len(self.dataloader)
    
    
    
class TestDataset_adv(torch.utils.data.Dataset):
    def __init__(self, userInput, itemInput, labels):
        super(TestDataset_adv, self).__init__()
        self.userInput = torch.Tensor(userInput).long()
        self.itemInput = torch.Tensor(itemInput).long()
        self.labels = torch.Tensor(labels)

    def __getitem__(self, index):
        return self.userInput[index], self.itemInput[index], self.labels[index]

    def __len__(self):
        return self.userInput.size(0)
    
    
    
    

    
def create_logger(log_path):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger

def create_scheduler(args, optimizer, lr_decays=None):
    
	if args.lr_scheduler == "step":
		if lr_decays is None:
			lr_decays = [int(args.epochs * 0.5), int(args.epochs * 0.75)]
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decays, gamma=args.lr_decay_gamma, last_epoch=-1)
	elif args.lr_scheduler == "cosine":
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
	else:
		raise ValueError("The scheduler is not implemented!")
	# elif args.lr_scheduler == "cyclic":
	# 	pass
	return scheduler




def evalulate_robustness(args, model, ldr, opt, epsilon, testRatings, testNegatives, topK, topK1, topK2, topK3):
    emb_backup = {}
    criterion = torch.nn.BCELoss()
    is_first_attack = True
    start_time = time.time()
    cloned_model = deepcopy(model)
    rounds = 10
    for name, param in cloned_model.named_parameters():
            param.requires_grad = True
            
    optimizer = torch.optim.SGD(cloned_model.parameters(), lr=1)

    cloned_model.train()
    if args.attack == 'fgsm':
        for ui, ii, lbl in ldr:
            optimizer.zero_grad()
            ui, ii, lbl = ui.to(device), ii.to(device), lbl.to(device)  
            loss1,*_,outputs= cloned_model(ui, ii,lbl)
            outputs = outputs.squeeze()
            loss =  criterion(outputs, lbl)
            loss.backward()
            
            with torch.no_grad():
                for name, param in cloned_model.named_parameters():
                    if is_first_attack:
                        emb_backup[name] = param.data.clone()
                    norm = torch.norm(param.grad)
                    if norm != 0 and not torch.isnan(norm):
                        r_at = param.grad / norm
                        param.data.add_(r_at)
                        r = param.data - emb_backup[name]
                        if torch.norm(r) > epsilon:
                            r = epsilon * r / torch.norm(r)
                        param.data = emb_backup[name] + r
                        
                        
                        # param.data = param.data + grads['1'].sign()*(self.epsilon/self.pro_num)
                        # delta = torch.clamp(y-self.y1, -self.epsilon, self.epsilon).to(self.device) 
                        # param.data = self.y1.add(delta).to(self.device) 
                is_first_attack = False
    # print(time.time() - start_time)
    
    if args.attack == 'pgd':
        for epoch in range(rounds):
                start_time = time.time()
                if is_first_attack:
                    # 备份原始模型参数
                    for name, param in cloned_model.named_parameters():
                        emb_backup[name] = param.data.clone()
                        if cloned_model and param.requires_grad:
                            # 初始化随机扰动
                            random_noise = epsilon * torch.randn_like(param)
                            param.data.add_(random_noise)
                            # 确保扰动不超过epsilon
                            param_diff = param.data - emb_backup[name]
                            if torch.norm(param_diff) > epsilon:
                                param.data = emb_backup[name] + epsilon * param_diff / torch.norm(param_diff)
                
                is_first_attack = False

                for ui, ii, lbl in ldr:
                    ui, ii, lbl = ui.to(device), ii.to(device), lbl.to(device)  
                    cloned_model.train()
                    loss,*_,output= cloned_model(ui, ii,lbl)
                    yc = output.squeeze()
                    loss = -criterion(yc, lbl)
                    opt.zero_grad()
                    loss.backward()
                    # 对模型参数应用扰动
                    with torch.no_grad():
                        for name, param in cloned_model.named_parameters():
                            if param.requires_grad:
                                # 计算扰动
                                norm = torch.norm(param.grad)
                                if norm != 0 and not torch.isnan(norm):
                                    # 添加扰动
                                    delta = epsilon/rounds * param.grad / norm
                                    param.data.add_(delta)
                                    # 确保扰动在epsilon范围内
                                    param_diff = param.data - emb_backup[name]
                                    if torch.norm(param_diff) > epsilon:
                                        param.data = emb_backup[name] + epsilon * param_diff / torch.norm(param_diff)
            
            # print(time.time()-start_time)
    
    
    cloned_model.eval()
    # 检查模型性能
    t1 = time.time()
    hits10, ndcgs10, maps10, mrrs10 = evaluate_model(cloned_model, testRatings, testNegatives, topK, topK1, topK2, topK3,attack1=None,epsilon1=None,alpha1=None,num_steps1=None,decay_factor=None, num_thread=1, device=device)

    # # 恢复模型的原始参数
    # with torch.no_grad():
    #     for name, param in model.named_parameters():
    #         if name in emb_backup:
    #             param.data = emb_backup[name]
    del cloned_model
    hr10, ndcg10,map10,mrr10 = np.array(hits10).mean(), np.array(ndcgs10).mean(),np.array(maps10).mean(), np.array(mrrs10).mean()

    # print(f"Epoch : HR10={hr10:.4f}, NDCG10={ndcg10:.4f}, mrrs10={mrr10:.4f}, HR20={hr20:.4f}, NDCG20={ndcg20:.4f}, mrrs20={mrr20:.4f}, HR50={hr50:.4f}, NDCG50={ndcg50:.4f}, mrrs50={mrr50:.4f} [{time.time()-t1:.1f}s]")
    return hr10





def interpolation(args, logger, init_sd, ft_sd, model, trainloader, criterion,save_dir):
    alphas = np.arange(0, 1.1, 0.1)

    # alphas = np.arange(0, 0.21, 0.01)
    
    records = []
    best_performance = 0
    best_checkpoint_path = None  # 在使用之前给变量一个初始值
    for alpha in alphas:
        model_dict = {}
        for name, _ in init_sd.items():
            model_dict[name] = alpha * ft_sd[name] + (1 - alpha) * init_sd[name]

        # 临时保存模型参数
        temp_checkpoint_path = save_dir + "finetune_{:.3f}_params.pth".format(alpha)
        torch.save(model_dict, temp_checkpoint_path)
        # torch.save(model_dict, save_dir + "finetune_{:.3f}_params.pth".format(alpha))

        model.load_state_dict(model_dict)
        hits10, ndcgs10, _, _, = evaluate_model(model, args.testRatings, args.testNegatives, 
                                                                         args.topK,args.topK1,args.topK2,args.topK3, num_thread=1, device=device)
        hr10 = np.array(hits10).mean()
        test_robust_acc = evalulate_robustness(args, model,trainloader,args.optimizer,args.epsilon,
                                                   args.testRatings, args.testNegatives, args.topK,args.topK1,args.topK2,args.topK3)

        logger.info("==> Alpha: {:.4f}, hits10: {:.4f}%, test robust hit10: {:.4f}%".format(alpha, hr10*100, test_robust_acc*100))
        records.append((hr10, test_robust_acc))

        # 使用hr10和test_robust_acc的综合值来评估性能，这里简单地取它们的平均值作为示例
        current_performance = (hr10 + test_robust_acc) / 2
        if alpha == 0:
            test_robust_acc_init = test_robust_acc
        # 更新最优模型


        if test_robust_acc >=test_robust_acc_init:
            if current_performance > best_performance:
                best_performance = current_performance
                best_alpha = alpha
                if best_checkpoint_path is not None and os.path.isfile(best_checkpoint_path):
                    try:
                        os.remove(best_checkpoint_path)
                    except Exception as e:
                        print(f"删除文件时出错：{e}")
                else:
                    print("未提供有效的 best_checkpoint_path,跳过删除步骤。")

                best_checkpoint_path = temp_checkpoint_path
            # 注意：这里我们保留最优模型的checkpoint路径，以便后续使用，不再删除
            
    # 循环结束后，载入最优模型参数
    if best_checkpoint_path:
        best_model_dict = torch.load(best_checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(best_model_dict)
        logger.info("Loaded best model with alpha {:.4f}".format(best_alpha))


    return records
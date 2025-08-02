
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from collections import defaultdict
import time
from matplotlib import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from copy import deepcopy
# from tqdm import tqdm
import logging
# from auto_attacks.eval import Normalize

import torch
import torch.optim as optim
import logging
from scipy.sparse import dok_matrix
# from dataloader import *
# from model import create_model
# from optimizer import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
from utils import *
from mlp import MLP
from Dataset import Dataset
from utils_awp_RI import AdvWeightPerturb1

def layer_sharpness(args, model, trainloader,epsilon=0.1):
    criterion = torch.nn.BCELoss()
    # 创建训练数据加载器
    # trainloader = torch.utils.data.DataLoader(generate_adv_dataset(args, deepcopy(model)), batch_size=512, shuffle=True, num_workers=0)
    origin_total = 0
    origin_loss = 0.0
    origin_loss111 = 0.0
    origin_acc = 0
    # 计算原始模型性能：包括未扰动时的损失origin_loss
    with torch.no_grad():
        model.eval()
        
        for ui, ii, lbl in trainloader:
            ui, ii, lbl = ui.to(device), ii.to(device), lbl.to(device)  

            # robust_output,_ = model(ui, ii)
            loss,_,_,_,_,_,_,_= model(ui, ii,lbl)
            origin_loss += loss
            
        # 分布式多GPU训练
        # origin_loss = origin_loss111[0] + origin_loss111[1]
    
    # 加一个测试模型性能
    # hit10 = evalulate_robustness(args, model,trainloader,args.optimizer,args.epsilon,args.testRatings,args.testNegatives, args.topK,args.topK1,args.topK2,args.topK3)
    
    # 1111111111111111111111111111111  1105撤回
    # 原始模型的性能指标 HR，NDCG
    hits10, ndcgs10, maps10, mrrs10 = evaluate_model(model, args.testRatings,args.testNegatives,  args.topK,args.topK1,args.topK2,args.topK3, num_thread=1, device=args.device)
    hit10, ndcg10,map10,mrr10 = np.array(hits10).mean(), np.array(ndcgs10).mean(),np.array(maps10).mean(), np.array(mrrs10).mean()
                
    # origin_acc /= origin_total

    # model.to('cpu')
    
    torch.cuda.empty_cache()  # 释放显存
    hit10_robust = evalulate_robustness(args, model,trainloader,args.optimizer,args.epsilon,args.testRatings,args.testNegatives, args.topK,args.topK1,args.topK2,args.topK3)
    origin_hit10 = hit10_robust
    origin_loss /= len(trainloader)
    # args.logger.info("{:35}, Robust Loss: {:10.4f}, Robust Acc: {:10.4f}".format("Origin", origin_loss, origin_acc*100))
    args.logger.info("{:35}, Robust Loss: {:10.4f}, Init hit10: {:10.4f},Init_Robust hit10: {:10.4f}".format("Origin", origin_loss,hit10*100, origin_hit10*100))
    # args.logger.info("{:35}, Robust Loss: {:10.4f}".format("Origin", origin_loss))
    model.eval()
    torch.cuda.empty_cache()  # 释放显存


    layer_sharpness_dict = {} 
    # 模型层遍历和锐度计算
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            # logger.info(name)
            # For WideResNet
            if "sub" in name:
                continue
            layer_sharpness_dict[name] = 1e10  # 这个大数值可以理解为一个占位符，后续的过程中会计算并更新每个层的实际锐度值。

    # 冻结其他层，只训练当前层
    for layer_name, _ in model.named_parameters():
        if "weight" in layer_name and layer_name[:-len(".weight")] in layer_sharpness_dict.keys():   # 选出那些名字中包含"weight"且在layer_sharpness_dict字典中有对应项的层
            # logger.info(layer_name)
            cloned_model = deepcopy(model)
            # set requires_grad sign for each layer
            for name, param in cloned_model.named_parameters():
                # logger.info(name)
                if name == layer_name:
                    # logger.info(name)
                    param.requires_grad = True
                    init_param = param.detach().clone()
                else:
                    param.requires_grad = False

            # 学习率为1
            optimizer = torch.optim.SGD(cloned_model.parameters(), lr=1)

            max_loss = 0.0
            min_acc = 0

            # 三次迭代训练当前层
            for epoch in range(3):   # 这里10改3 
                # Gradient ascent
                cloned_model.train()
                for ui, ii, lbl in trainloader:
                    optimizer.zero_grad()
                    ui, ii, lbl = ui.to(device), ii.to(device), lbl.to(device)  
                    loss1,_,_,_,_,_,_,outputs= cloned_model(ui, ii,lbl)
                    outputs = outputs.squeeze()
                    loss = - criterion(outputs, lbl)
                    loss.backward()
                    optimizer.step()
                    
                cloned_model.eval()
                # for inputs, targets in trainloader:
                #     optimizer.zero_grad()
                #     outputs = cloned_model(inputs)
                #     loss = -1 * criterion(outputs, targets) 
                #     loss.backward()
                #     optimizer.step()
                sd = cloned_model.state_dict()
                diff = sd[layer_name] - init_param
                times = torch.linalg.norm(diff)/torch.linalg.norm(init_param)
                # logger.info(times)

                # 把噪声大小规范为epsilon比例内

                # if times > epsilon:
                #     diff = diff / times * epsilon
                #     sd[layer_name] = deepcopy(init_param + diff)
                #     cloned_model.load_state_dict(sd)

                size_para = init_param.size(0)
                normVal1 = torch.norm(diff.view(size_para, -1), 2, 1)
                normVal2 = torch.norm(init_param.view(size_para, -1), 2, 1)
                # mask = normVal1<=normVal2 * self.epsilon
                scaling = normVal2/normVal1 * epsilon   # 算出一次噪声需要调整的倍数
                scaling[scaling == float('inf')] = 0
                # scaling[mask] = 1
                diff = diff*scaling.view(size_para, 1)
                #param.data = param.data + diff # 把这一次的噪声加到参数上、

                sd[layer_name] = deepcopy(init_param + diff)
                cloned_model.load_state_dict(sd)
                torch.cuda.empty_cache()  # 释放显存


                # 计算扰动后该层导致的系统整体损失
                with torch.no_grad():
                    total = 0
                    total_loss = 0.0
                    total_loss1 = 0.0
                    correct = 0
                    for ui, ii, lbl in trainloader:
                        ui, ii, lbl = ui.to(device), ii.to(device), lbl.to(device)  
                        loss1,_,_,_,_,_,_,outputs= cloned_model(ui, ii,lbl)
                        total_loss += loss1
                        
                    # for inputs, targets in trainloader:
                    #     outputs = cloned_model(inputs)
                    #     total += targets.shape[0]
                    #     total_loss += criterion(outputs, targets).item() * targets.shape[0]
                    #     _, predicted = outputs.max(1)
                    #     correct += predicted.eq(targets).sum().item()  
                    # total_loss = total_loss1[0] + total_loss1[1]
                    total_loss /= len(trainloader)
                    # correct /= total
                # 加一个测试模型性能
                hit10 = evalulate_robustness(args, cloned_model,trainloader,args.optimizer,args.epsilon,args.testRatings,args.testNegatives, args.topK,args.topK1,args.topK2,args.topK3)
            #     hits10, ndcgs10, maps10, mrrs10, hits20, ndcgs20,maps20, mrrs20,hits50, ndcgs50,maps50, mrrs50 = evalulate_robustness(cloned_model, args.testRatings,args.testNegatives,  args.topK,args.topK1,args.topK2,args.topK3, num_thread=1, device=args.device)
            #     hit10, ndcg10,map10,mrr10, hr20, ndcg20,map20,mrr20,hr50, ndcg50,map50, mrr50 = np.array(hits10).mean(), np.array(ndcgs10).mean(),np.array(maps10).mean(), np.array(mrrs10).mean(),\
            # np.array(hits20).mean(), np.array(ndcgs20).mean(), np.array(maps20).mean(), np.array(mrrs20).mean(), np.array(hits50).mean(), np.array(ndcgs50).mean(), np.array(maps50).mean(), np.array(mrrs50).mean()
                
                if total_loss > max_loss:
                    max_loss = total_loss
                    min_hit10 = hit10
            del cloned_model
            
            # MRC计算，用扰动后的最大损失减去扰动前的损失
            layer_sharpness_dict[layer_name[:-len(".weight")]] = max_loss - origin_loss
            # args.logger.info("{:35}, MRC: {:10.4f}, Dropped Robust Acc: {:10.4f}".format(layer_name[:-len(".weight")], max_loss-origin_loss, (origin_acc-min_acc)*100))
            args.logger.info("{:35}, MRC: {:10.4f}, Dropped Robust hit10: {:10.4f}".format(layer_name[:-len(".weight")], 
                                                                                         max_loss-origin_loss, (min_hit10-origin_hit10)*100))
            # args.logger.info("{:35}, MRC: {:10.4f}".format(layer_name[:-len(".weight")], max_loss-origin_loss,(origin_acc-min_acc)*100))

    # 排序MRC值
    sorted_layer_sharpness = sorted(layer_sharpness_dict.items(), key=lambda x:x[1])
    for (k, v) in sorted_layer_sharpness:
        args.logger.info("{:35}, Robust Loss: {:10.4f}".format(k, v))
    
    return sorted_layer_sharpness


def train(args, model, dataloader, optimizer, criterion):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    start_time = time.time()
    # Generate training instances
    # userInput, itemInput, labels = get_train_instances(train, args.nNeg)
    # dst = BatchDataset(userInput, itemInput, labels)
    # ldr = torch.utils.data.DataLoader(dst, batch_size=args.batch_size, shuffle=True, drop_last=True)
    losses = AverageMeter("Loss")

    for ui, ii, lbl in dataloader:
        ui, ii, lbl = ui.to(device), ii.to(device), lbl.to(device)  

        # 计算对抗性权重扰动，并将其添加到模型权重中
        awp = args.awp_adversary.calc_awp(inputs_adv_u=ui, inputs_adv_i=ii,inputs_adv_lab=lbl,layers = args.layer)
        args.awp_adversary.perturb(awp)

        loss,_,_,_,_,_,_,_= model(ui, ii,lbl)

        # 反向传播与参数更新，并计算总损失
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        args.awp_adversary.restore(awp)
        

    return train_loss / len(dataloader)


def main():

    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--model_path', default="pretrained/lastfm_MLP_init_1741682452.2276385.pth", type=str, help='model_path')  ##########
    parser.add_argument('--model', default="RAWP", type=str, help='model used')   
    parser.add_argument("--data_path", nargs="?", default="Data/",
                        help="Input data path.")
    parser.add_argument("--dataset", nargs="?", default="lastfm",   # ml-1m yelp  AToy lastfm #####################
                        help="Choose a dataset.")
    parser.add_argument('--num_classes', default=10, type=int, help='num classes')
    parser.add_argument('--layer', default=None, type=str, help='Trainable layer') # True
    parser.add_argument("--cal_mrc", default=True,action="store_true", help='If to calculate Module Robust Criticality (MRC) value of each module.')
    parser.add_argument('--epsilon', default=0.008, type=int)           # 检测鲁棒性的噪声扰动
    parser.add_argument("--alpha", nargs="?", default=0.005/10)  # 0.5/25
    parser.add_argument("--num_steps", nargs="?", default=10)  #25
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate') #####################################
    parser.add_argument('--resume', default=None, type=str, help='resume from checkpoint')
    parser.add_argument('--lr-prox', default=0, type=float)
    parser.add_argument('--awp-gamma', default=0.002, type=float) # 0  0.01#################################
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--patch', default=4, type=int, help='num patch (used by vit)')
    parser.add_argument('--optim', default="SGDM", type=str, help="optimizer")
    parser.add_argument('--attack', default='fgsm', type=str, choices=['pgd', 'fgsm', 'bim', 'none'])
    parser.add_argument('--device', default="cuda", type=str, help='device')
    parser.add_argument('--lr_scheduler', default="step", choices=["step", 'cosine'])
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum for SGDM')
    parser.add_argument('--lr_decay_gamma', default=0.1, type=float, help='lr_decay_gamma')
    parser.add_argument('--wd', default=0.0005, type=float, help='weight decay')
    parser.add_argument('--epochs', default=20
                        , type=int, help='num of epochs')
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')
    parser.add_argument("--fcLayers", nargs="?", default="[1024, 512, 256, 128, 64, 32, 16]", #  [512, 256, 128, 64, 32, 16]  [512,  128, 32]
                        help="Size of each layer. Note that the first layer is the "
                             "concatenation of user and item embeddings. So fcLayers[0]/2 is the embedding size.")
    parser.add_argument("--nNeg", type=int, default=4,help="Number of negative instances to pair with a positive instance.")

    args = parser.parse_args()
    device = args.device

    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)

    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    
    All_start_time = time.time()
    fcLayers = eval(args.fcLayers)
    topK = 10
    topK1 = 20
    topK2 = 50
    topK3 = 100
    
    # Loading data
    t1 = time.time()
    dataset = Dataset(args.data_path + args.dataset)
    train_data, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    nUsers, nItems = train_data.shape
    
    arr = np.array(testRatings)
    max_value_second_column1 = np.max(arr[:, 1])
    arr = np.array(testNegatives)
    max_value_second_column2 = np.max(arr)
    n_value = max(max_value_second_column1, max_value_second_column2)
    if nItems <= n_value:
        nItems = n_value+1
        extended_sparse_matrix = dok_matrix((nUsers, nItems), dtype=np.float32)

        for (row, col), value in train_data.items():
            extended_sparse_matrix[row, col] = value
            train_data = extended_sparse_matrix
    
    
    userMatrix = torch.Tensor(get_train_matrix(train_data))
    itemMatrix = torch.transpose(torch.Tensor(get_train_matrix(train_data)), 0, 1)
    # print(f"userType{type(userMatrix)}, itemType{type(itemMatrix)}")
    userMatrix, itemMatrix = userMatrix.to(device), itemMatrix.to(device)
    
    # userMatrix, itemMatrix = userMatrix.to(args.device), itemMatrix.to(args.device)
    print(f"Load data: #user={nUsers}, #item={nItems}, #train_data={train_data.nnz}, #test={len(testRatings)} [{time.time()-t1:.1f}s]")

    print("RAWP-weight:*************************************")
    if args.model == 'RAWP':
        model = MLP(fcLayers, userMatrix, itemMatrix,args.device)
        proxy_adv = MLP(fcLayers, userMatrix, itemMatrix,args.device)
    else:
        raise ValueError("Unknown model")

    model = nn.DataParallel(model).to(args.device)
    proxy_adv = nn.DataParallel(proxy_adv).to(device)  # 1105
    params = model.parameters()
    userInput, itemInput, labels = get_train_instances(train_data, args.nNeg)


    dst = BatchDataset(userInput, itemInput, labels)
    trainloader = torch.utils.data.DataLoader(dst, batch_size=args.batch_size, shuffle=True, drop_last=True)
    

    proj_name = "rift_refine_layer_robust_lastfm"
    best_hr10 = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    suffix = '{}_{}_lr={}_g={}_wd={}_epochs={}_{}'.format(proj_name, args.optim, args.lr,args.awp_gamma,  args.wd, args.epochs, args.layer)
    model_save_dir = './results/MLP_clean/FT/{}_{}/checkpoint/'.format(args.model, args.dataset) + suffix + "/"

    for path in [model_save_dir]:
        if not os.path.isdir(path):
            os.makedirs(path)

    logger = create_logger(model_save_dir+'output.log')
    logger.info(args)

    args.logger = logger

  
    logger.info('==> Building dataloaders...')
    logger.info(args.dataset)

    # create model
    logger.info('==> Building model...')
    # 加载预训练的参数 

    model_path = args.model_path
    logger.info(model_path) 
    if 'robust' in model_path:  
        # 假设 original_state_dict 是你加载的原始状态字典 
        original_state_dict = torch.load(model_path,map_location=device, weights_only=True)
        # # 如果是mlp生成的模型，有net需要提取出来
        # original_state_dict = original_state_dict['net']
        
        # 修改键以匹配 DataParallel 的期望格式
        modified_state_dict = {'module.' + key: value for key, value in original_state_dict.items()}
        
        # 创建一个新的state_dict，只包含不带有'reg'字段的参数
        new_state_dict111 = {}
        for key, value in modified_state_dict.items():
            if 'reg' not in key:
                new_state_dict111[key] = value
        model.load_state_dict(new_state_dict111)
        
        # # 现在使用修改后的状态字典加载模型
        # model.load_state_dict(modified_state_dict) 
    else:  # RAWP
        original_state_dict = torch.load(model_path,map_location=device, weights_only=True)
        # 创建一个新的state_dict，只包含不带有'reg'字段的参数
        new_state_dict111 = {}
        for key, value in original_state_dict.items():
            if 'reg' not in key:
                new_state_dict111[key] = value
        model.load_state_dict(new_state_dict111)


    model.load_state_dict(new_state_dict111) # 
    del new_state_dict111
    torch.cuda.empty_cache()
    del original_state_dict
    torch.cuda.empty_cache()  # 释放显存

    logger.info(args.model)

    logger.info('==> Building optimizer and learning rate scheduler...')
    optimizer = torch.optim.SGD(params, lr=args.lr,momentum = args.momentum, weight_decay=args.wd)
    proxy_opt = torch.optim.Adam(proxy_adv.parameters(), lr=args.lr_prox)   # 0.01  # 1105
    args.awp_adversary = AdvWeightPerturb1(model=model, proxy=proxy_adv, proxy_optim=proxy_opt, gamma=args.awp_gamma) # 1105
    criterion = torch.nn.BCELoss()



    logger.info(optimizer)
    lr_decays = [int(args.epochs // 2)]
    scheduler = create_scheduler(args, optimizer, lr_decays=lr_decays)
    logger.info(scheduler)


    init_sd = deepcopy(model.state_dict())  # 保存模型的初始状态：
    torch.save(init_sd, model_save_dir + "init_params.pth")


    # init_sd = init_sd.to('cpu')
    for key, value in init_sd.items():
        if hasattr(value, 'to'):
            init_sd[key] = value.to('cpu')
        else:
            init_sd[key] = value
    torch.cuda.empty_cache()


    # init_sd = deepcopy(model.state_dict())  # 保存模型的初始状态
    # init_sd.load_state_dict(model_save_dir + "init_params.pth")

    args.trainloader = trainloader
    args.optimizer = optimizer
    # args.epsilon = epsilon
    args.testRatings = testRatings
    args.testNegatives = testNegatives
    args.topK = topK
    args.topK1 = topK1
    args.topK2 = topK2
    args.topK3 = topK3
    for rounds_1 in range(1):
        if args.cal_mrc:  #  计算模型层的锐度：    33###################################################################################
            # sorted_layer_sharpness = layer_sharpness(args, deepcopy(model),trainloader, args.epsilon)
            sorted_layer_sharpness = layer_sharpness(args, model,trainloader, args.epsilon)
            # exit()
        
            # 使用min函数找到最小值对应的层名称
            min_layer_name = sorted_layer_sharpness[8][0]
            args.layer = min_layer_name

        logger.info(args.layer)
        assert args.layer is not None # 这一行确保了通过命令行参数指定了某一层（可能是关注的焦点），
                                        #用于后续的训练或分析。如果未指定任何层，则程序将因断言失败而提前终止。

        for name, param in model.named_parameters():  # 设置模型参数的requires_grad属性：
            param.requires_grad = False
            if args.layer in name:
                param.requires_grad = True  # 包含args.layer指定的字符串，那么将这个参数的requires_grad属性设置为True。这表示只有特定的层（或层的一部分）将在训练过程中被更新。

        hits10, ndcgs10, maps10, mrrs10 = evaluate_model(model, testRatings, testNegatives, topK,topK1,topK2,topK3, num_thread=1, device=args.device)
        hr10, ndcg10,map10,mrr10 = np.array(hits10).mean(), np.array(ndcgs10).mean(),np.array(maps10).mean(), np.array(mrrs10).mean()
        logger.info(f"init: HR10={hr10:.4f}, NDCG10={ndcg10:.4f}, mrrs10={mrr10:.4f}")
            

        # logging.info(downsampling_ratio)

        # _, train_acc = evaluate(args, model, trainloader, criterion)  # 评估模型的初始性能：
        # _, test_acc = evaluate(args, model, testloader, criterion)  # 
        test_robust_hit10 = evalulate_robustness(args, model,trainloader,optimizer,args.epsilon,testRatings, testNegatives, topK,topK1,topK2,topK3)  # 评估模型在对抗性数据集上的鲁棒性（test_robust_hit10）
        # 记录初始性能指标：
        logger.info("==> Init  hits10:: {:.4f}%, robust hit10: {:.4f}%".format(hr10*100, test_robust_hit10*100))
        original_hr_total = hr10 + test_robust_hit10
        label_train_true = 0

        args.init_HR10 = hr10

        for epoch in range(start_epoch, start_epoch + args.epochs):
            # logging.info('Downsampling ratio: {}'.format(downsampling_ratio))
            logger.info("==> Epoch {}".format(epoch))
            logger.info("==> Training...")
            train_loss = train(args, model, trainloader, optimizer, criterion)

            logger.info("==> Train loss: {:.4f}".format(train_loss))

            logger.info("==> Testing...")
            # # 提精度
            hits10, ndcgs10, maps10, mrrs10 = evaluate_model(model, testRatings, testNegatives, topK,topK1,topK2,topK3, num_thread=1, device=device)
            # test_loss, test_acc = evaluate_model(args, model, testloader, criterion)
            hr10_temp_befor_attack, ndcg10,map10,mrr10 = np.array(hits10).mean(), np.array(ndcgs10).mean(),np.array(maps10).mean(), np.array(mrrs10).mean()
            # 提鲁棒性
            hr10 = evalulate_robustness(args, model,trainloader,args.optimizer,args.epsilon,args.testRatings,args.testNegatives, args.topK,args.topK1,args.topK2,args.topK3)
            
            logger.info("==> hits10: {:.4f}%  robust_hits10: {:.4f}%".format(hr10_temp_befor_attack*100 , hr10*100))

            state = {
                'model': model.state_dict(),
                'acc': hr10,
                'epoch': epoch,
            }
            
            # 测试一下攻击前的鲁棒性然后与鲁棒hr相加判断是否要加到最优里面
            temp_hr__total = hr10_temp_befor_attack +  hr10
            
            # 保存最佳模型
            if hr10 > best_hr10:
            # if original_hr_total < temp_hr__total:
                best_hr10 = hr10
                # original_hr_total = temp_hr__total
                params = "best_params.pth"
                logger.info('==> Saving best params...')
                torch.save(state, model_save_dir + params)
                label_train_true = 1
            # else:
            #     if epoch % 2 == 0:
            #         params = "epoch{}_params.pth".format(epoch)
            #         logger.info('==> Saving checkpoints...')
            #         torch.save(state, model_save_dir + params)

            scheduler.step()

        del proxy_adv
        torch.cuda.empty_cache()  # 释放显存
        if(label_train_true > 0 ):
            checkpoint = torch.load(model_save_dir + "best_params.pth", weights_only=False)
            model.load_state_dict(checkpoint["model"])
            del checkpoint
            torch.cuda.empty_cache()  # 释放显存
        
        
        
        model_copy = deepcopy(model.state_dict())
        for key, value in model_copy.items():
            if hasattr(value, 'to'):
                model_copy[key] = value.to('cpu')
            else:
                model_copy[key] = value
        torch.cuda.empty_cache()  # 释放显存

        # test_loss, test_acc = evaluate(args, model, testloader, criterion)
        hits10, ndcgs10, maps10, mrrs10 = evaluate_model(model, testRatings, testNegatives, topK,topK1,topK2,topK3, num_thread=1, device=device)
        hr10, ndcg10,map10,mrr10 = np.array(hits10).mean(), np.array(ndcgs10).mean(),np.array(maps10).mean(), np.array(mrrs10).mean()
        test_robust_hit10 = evalulate_robustness(args, model,trainloader,optimizer,args.epsilon,testRatings, testNegatives, topK,topK1,topK2,topK3)

        logger.info("==> Finetune hits10: {:.4f}%, robust hit10: {:.4f}".format(hr10*100, test_robust_hit10*100))

        # 线性插值更新特定层
        record = interpolation(args, logger, init_sd, model_copy , model, trainloader, criterion, model_save_dir)
        logger.info(record)
        
        modelPath = f"pretrained/{args.dataset}_{args.lr}_{args.awp_gamma}_FT_Inter.pth"
        os.makedirs("pretrained", exist_ok=True)
        torch.save(model.state_dict(), modelPath)

        # init_sd = deepcopy(model.state_dict())

        logger.info("save:=================================")
        logger.info(modelPath)
        logger.info(time.time()-All_start_time)

if __name__ == "__main__":
    main()







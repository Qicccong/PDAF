
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from collections import defaultdict
import random
import time
from matplotlib import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from copy import deepcopy
import torch
import torch.optim as optim
import logging
from scipy.sparse import dok_matrix

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
from utils import *
from mlp import MLP
from Dataset import Dataset
from utils_awp_RI import AdvWeightPerturb1


def layer_sharpness(args, model, trainloader,epsilon=0.1):
    criterion = torch.nn.BCELoss()
    num_tcal = 3000   # 
    # 创建训练数据加载器
    origin_loss = 0.0
    # 计算原始模型性能：
    with torch.no_grad():
        model.eval()
        cal_num = 0
        for ui, ii, lbl in trainloader:
            ui, ii, lbl = ui.to(device), ii.to(device), lbl.to(device)  
            loss,_,_,_,_,_,_,_= model(ui, ii,lbl)
            origin_loss += loss
            cal_num = cal_num + 1
            if  cal_num >= num_tcal:
                    break
            
    hits10, ndcgs10, maps10, mrrs10 = evaluate_model(model, args.testRatings,args.testNegatives,  args.topK,args.topK1,args.topK2,args.topK3, num_thread=1, device=args.device)
    hit10, ndcg10,map10,mrr10 = np.array(hits10).mean(), np.array(ndcgs10).mean(),np.array(maps10).mean(), np.array(mrrs10).mean()
 
    torch.cuda.empty_cache()  # 释放显存
    args.logger.info("{:35}, Origin Loss: {:10.4f}, Init hit10: {:10.4f}".format("Origin", origin_loss,hit10*100))
    model.eval()
    torch.cuda.empty_cache()  # 释放显存
    layer_sharpness_dict = {} 
    # 模型层遍历和锐度计算
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if "sub" in name:
                continue
            layer_sharpness_dict[name] = 1e10  # 这个大数值可以理解为一个占位符，后续的过程中会计算并更新每个层的实际锐度值。

    for layer_name, _ in model.named_parameters():
        if "weight" in layer_name and layer_name[:-len(".weight")] in layer_sharpness_dict.keys():   # 选出那些名字中包含"weight"且在layer_sharpness_dict字典中有对应项的层
            cloned_model = deepcopy(model)
            for name, param in cloned_model.named_parameters():
                if name == layer_name:
                    param.requires_grad = True
                    init_param = param.detach().clone()
                else:
                    param.requires_grad = False
        
            optimizer = torch.optim.SGD(cloned_model.parameters(), lr=1)

            max_loss = 0.0
            min_acc = 0
            cal_num =0
            for epoch in range(1):   # 这里10改3 
                # Gradient ascent   对抗攻击
                cloned_model.train()
                for ui, ii, lbl in trainloader:
                    optimizer.zero_grad()
                    ui, ii, lbl = ui.to(device), ii.to(device), lbl.to(device)  
                    loss1,_,_,_,_,_,_,outputs= cloned_model(ui, ii,lbl)
                    outputs = outputs.squeeze()
                    loss = - criterion(outputs, lbl)
                    loss.backward()
                    optimizer.step()
                    cal_num = cal_num +1
                    # if  cal_num >= num_tcal:
                    #         break
                cloned_model.eval()
                sd = cloned_model.state_dict()
                diff = sd[layer_name] - init_param
                times = torch.linalg.norm(diff)/torch.linalg.norm(init_param)


                size_para = init_param.size(0)
                normVal1 = torch.norm(diff.view(size_para, -1), 2, 1)
                normVal2 = torch.norm(init_param.view(size_para, -1), 2, 1)
                scaling = normVal2/normVal1 * epsilon   # 算出一次噪声需要调整的倍数
                scaling[scaling == float('inf')] = 0
                diff = diff*scaling.view(size_para, 1)

                sd[layer_name] = deepcopy(init_param + diff)
                cloned_model.load_state_dict(sd)
                torch.cuda.empty_cache()  # 释放显存

                # 检测被攻击后模型的损失
                with torch.no_grad():
                    total = 0
                    total_loss = 0.0
                    total_loss1 = 0.0
                    correct = 0
                    cal_num = 0
                    for ui, ii, lbl in trainloader:
                        ui, ii, lbl = ui.to(device), ii.to(device), lbl.to(device)  
                        loss1,_,_,_,_,_,_,outputs= cloned_model(ui, ii,lbl)
                        total_loss += loss1
                        cal_num = cal_num +1
                        if cal_num >= num_tcal:
                            break
                    
                    
                if total_loss > max_loss:
                    max_loss = total_loss
                    
            del cloned_model
            layer_sharpness_dict[layer_name[:-len(".weight")]] = max_loss - origin_loss
            args.logger.info("{:35}, MRC: {:10.4f}".format(layer_name[:-len(".weight")], max_loss-origin_loss))

    return layer_sharpness_dict




def train(args, model, dataloader, optimizer, criterion):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    start_time = time.time()
    losses = AverageMeter("Loss")

    for ui, ii, lbl in dataloader:
        ui, ii, lbl = ui.to(device), ii.to(device), lbl.to(device)  

        awp = args.awp_adversary.calc_awp(inputs_adv_u=ui, inputs_adv_i=ii,inputs_adv_lab=lbl,layers = args.layer)
        args.awp_adversary.perturb(awp)

        loss,_,_,_,_,_,_,_= model(ui, ii,lbl)


        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        args.awp_adversary.restore(awp)
        
    return train_loss / len(dataloader)


def main():
        # pretrained/80_ml-1m_MLP_init_1713513910.3174512.pth   pretrained/79_lastfm_MLP_RAWP_1718355934.5297291.pth
    parser = argparse.ArgumentParser(description='PyTorch Training')  
    parser.add_argument('--model_path', default="pretrained/80_ml-1m_MLP_init_1713513910.3174512.pth", type=str, help='model_path')
    parser.add_argument('--model', default="RAWP", type=str, help='model used')
    parser.add_argument("--data_path", nargs="?", default="Data/",
                        help="Input data path.")
    parser.add_argument("--dataset", nargs="?", default="ml-1m",   # ml-1m yelp  AToy lastfm #####################
                        help="Choose a dataset.")
    parser.add_argument('--num_classes', default=10, type=int, help='num classes')
    parser.add_argument('--layer', default=None, type=str, help='Trainable layer') # True
    parser.add_argument("--cal_mrc", default=True,action="store_true", help='If to calculate Module Robust Criticality (MRC) value of each module.')
    parser.add_argument('--epsilon', default=0.008, type=float)
    parser.add_argument("--alpha", nargs="?", default=0.008/10)  # 0.5/25
    parser.add_argument("--num_steps", nargs="?", default=10)  #25
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate') 
    parser.add_argument('--resume', default=None, type=str, help='resume from checkpoint')
    parser.add_argument('--lr-prox', default=0, type=float)
    parser.add_argument('--awp-gamma', default=0.008, type=float) # 0  0.01#################################
    parser.add_argument('--seed', default=2, type=int)
    parser.add_argument('--patch', default=4, type=int, help='num patch (used by vit)')
    parser.add_argument('--optim', default="SGDM", type=str, help="optimizer")
    parser.add_argument('--attack', default='fgsm', type=str, choices=['pgd', 'fgsm', 'bim', 'none'])
    parser.add_argument('--device', default="cuda", type=str, help='device')
    parser.add_argument('--lr_scheduler', default="step", choices=["step", 'cosine'])
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum for SGDM')
    parser.add_argument('--lr_decay_gamma', default=0.1, type=float, help='lr_decay_gamma')
    parser.add_argument('--wd', default=0.0005, type=float, help='weight decay')
    parser.add_argument('--rounds', default=5, type=int, help='rounds of fine-tune')
    parser.add_argument('--epochs', default=5   # 多少个epoch检测一次模型最脆弱层次
                        , type=int, help='num of epochs')
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')
    parser.add_argument("--fcLayers", nargs="?", default="[1024, 512, 256, 128, 64, 32, 16]", #  [512, 256, 128, 64, 32, 16]  [512,  128, 32]
                        help="Size of each layer. Note that the first layer is the "
                             "concatenation of user and item embeddings. So fcLayers[0]/2 is the embedding size.")
    parser.add_argument("--nNeg", type=int, default=4,help="Number of negative instances to pair with a positive instance.")
    parser.add_argument('--proj_name', default=f"FTS_robust_{parser.parse_args().model}_80_qc", type=str, help='model used')

    args = parser.parse_args()
    device = args.device
    
    random_number = random.random()
    print(random_number)

    
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
    userMatrix, itemMatrix = userMatrix.to(args.device), itemMatrix.to(args.device)
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
    

    proj_name = args.proj_name
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    suffix = '{}_{}_lr={}_g={}_wd={}_epochs={}'.format(proj_name, args.optim, args.lr, args.awp_gamma ,args.wd, args.epochs)
    model_save_dir = './results/MLP_clean/PDAF/{}_{}/checkpoint/'.format(args.model, args.dataset) + suffix + "/"

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
    original_state_dict = torch.load(model_path, weights_only=False)
    # 创建一个新的state_dict，只包含不带有'reg'字段的参数
    new_state_dict111 = {}
    for key, value in original_state_dict.items():
        if 'reg' not in key:
            new_state_dict111[key] = value
            
    # # 修改键以匹配 DataParallel 的期望格式  RAT
    # modified_state_dict = {'module.' + key: value for key, value in new_state_dict111.items()}
    # 现在使用修改后的状态字典加载模型
    # model.load_state_dict(modified_state_dict) 


    model.load_state_dict(new_state_dict111) # 
    del new_state_dict111
    del original_state_dict
    torch.cuda.empty_cache()  # 释放显存

    logger.info(args.model)
    logger.info('==> Building optimizer and learning rate scheduler...')
    # optimizer = create_optimizer(args.optim, model, args.lr, args.momentum, weight_decay=args.wd)
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

    for key, value in init_sd.items():
        if hasattr(value, 'to'):
            init_sd[key] = value.to('cpu')
        else:
            init_sd[key] = value
    torch.cuda.empty_cache()

    args.trainloader = trainloader
    args.optimizer = optimizer
    # args.epsilon = epsilon
    args.testRatings = testRatings
    args.testNegatives = testNegatives
    args.topK = topK
    args.topK1 = topK1
    args.topK2 = topK2
    args.topK3 = topK3
    best_hr10 = 0  # best test accuracy

    for rounds_1 in range(5):
        args.rounds_1 = rounds_1
        
        if args.cal_mrc:  #  计算模型层的锐度：   
                sorted_layer_sharpness = layer_sharpness(args, model,trainloader, args.epsilon)
                sorted_items = sorted(sorted_layer_sharpness.items(), key=lambda x: x[1].item())
                min_layer_name = sorted_items[8][0]
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
        if rounds_1 == 0:
            args.init_HR10 = hr10

        test_robust_hit10 = evalulate_robustness(args, model,trainloader,optimizer,args.epsilon,testRatings, testNegatives, topK,topK1,topK2,topK3)  # 评估模型在对抗性数据集上的鲁棒性（test_robust_hit10）
        if rounds_1 == 0:
            best_hr10 = test_robust_hit10   # 训练前的鲁棒性能
            # best_hr10 = test_robust_hit10 + hr10  # 训练前的综合性能

        # 记录初始性能指标：
        logger.info("==> Init hits10:: {:.4f}%, robust hit10: {:.4f}%".format(hr10, test_robust_hit10))
        original_hr_total = hr10 + test_robust_hit10    # 训练前的综合性能
        label_train_true = 0


        # 保存训练前的初始模型
        state = {
            'model': model.state_dict(),
            'acc': test_robust_hit10,
            # 'epoch': epoch,   # 没有epoch信息
        }
        params = f"round_init_params.pth"
        logger.info('==> Saving round init params...')
        torch.save(state, model_save_dir + params)

        gap_gamma = args.awp_gamma / (args.epochs - 3)
        temp_awp_gamma = 3 * args.awp_gamma / args.epochs    # 从 1/3开始


        for epoch in range(start_epoch, start_epoch + args.epochs):

            # logging.info('Downsampling ratio: {}'.format(downsampling_ratio))
            logger.info("==> Epoch {}".format(epoch))
            logger.info("==> Training...")
            
            temp_awp_gamma = args.awp_gamma*(epoch+1)/args.epochs 
            # temp_awp_gamma += gap_gamma


            logger.info("==> gamma: {}".format(temp_awp_gamma))

            args.awp_adversary = AdvWeightPerturb1(model=model, proxy=proxy_adv, proxy_optim=proxy_opt, gamma=temp_awp_gamma)
            print(args.awp_adversary.gamma)
            train_loss = train(args, model, trainloader, optimizer, criterion)

            logger.info("==> Train loss: {:.4f}".format(train_loss))

            logger.info("==> Testing...")
            # # 提精度
            hits10, ndcgs10, maps10, mrrs10 = evaluate_model(model, testRatings, testNegatives, topK,topK1,topK2,topK3, num_thread=1, device=device)
            # test_loss, test_acc = evaluate_model(args, model, testloader, criterion)
            hr10_temp_befor_attack, ndcg10,map10,mrr10= np.array(hits10).mean(), np.array(ndcgs10).mean(),np.array(maps10).mean(), np.array(mrrs10).mean()
            
            # 提鲁棒性
            hr10_robust = evalulate_robustness(args, model,trainloader,args.optimizer,args.epsilon,args.testRatings,args.testNegatives, args.topK,args.topK1,args.topK2,args.topK3)
            logger.info("==> hits10: {:.4f}%  robust_hits10: {:.4f}%".format(hr10_temp_befor_attack*100 , hr10_robust*100))

            state = {
                'model': model.state_dict(),
                'acc': hr10_robust,
                'epoch': epoch,
            }
            
            # 测试一下攻击前的鲁棒性然后与鲁棒hr相加判断是否要加到最优里面
            temp_hr__total = hr10_temp_befor_attack +  hr10_robust  # 训练后的精度+鲁棒性能
            
            if hr10_robust >= best_hr10:    # 比较鲁棒性能

                best_hr10 = hr10_robust    # 存鲁棒性能
                params = f"best_params.pth"
                logger.info('==> Saving best params...')
                torch.save(state, model_save_dir + params)
                label_train_true = 1

            scheduler.step()

        
        torch.cuda.empty_cache()  # 释放显存
        # 该轮微调过程中出现了更优模型
        if(label_train_true > 0 ):
            checkpoint = torch.load(model_save_dir + f"best_params.pth", weights_only=False)
            model.load_state_dict(checkpoint["model"])
            del checkpoint
            # rounds_1 += 1
        else:   # 没有更优模型，就取训练前的模型，等于跳过一轮微调
            logger.info("No best_params.pth; load round_init_params.pth")
            checkpoint = torch.load(model_save_dir + f"round_init_params.pth", weights_only=False)
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

        hits10, ndcgs10, maps10, mrrs10 = evaluate_model(model, testRatings, testNegatives, topK,topK1,topK2,topK3, num_thread=1, device=device)
        hr10, ndcg10,map10,mrr10 = np.array(hits10).mean(), np.array(ndcgs10).mean(),np.array(maps10).mean(), np.array(mrrs10).mean()
        test_robust_hit10 = evalulate_robustness(args, model,trainloader,optimizer,args.epsilon,testRatings, testNegatives, topK,topK1,topK2,topK3)

        logger.info("==> Finetune hits10: {:.4f}%, robust hit10: {:.4f}".format(hr10*100, test_robust_hit10*100))
        
    # 插值部分
    record = interpolation(args, logger, init_sd, model_copy , model, trainloader, criterion, model_save_dir)
    logger.info(record)
            
    modelPath = f"pretrained/{args.dataset}_{args.lr}_{args.awp_gamma}_PDAF_Inter.pth"
    os.makedirs("pretrained", exist_ok=True)
    torch.save(model.state_dict(), modelPath)

    hits10, ndcgs10, maps10, mrrs10 = evaluate_model(model, testRatings, testNegatives, topK,topK1,topK2,topK3, num_thread=1, device=args.device)
    hr10, ndcg10,map10,mrr10 = np.array(hits10).mean(), np.array(ndcgs10).mean(),np.array(maps10).mean(), np.array(mrrs10).mean()
             

    for key, value in init_sd.items():
        if hasattr(value, 'to'):
            init_sd[key] = value.to('cpu')
        else:
            init_sd[key] = value
    torch.cuda.empty_cache()

    logger.info("save:=================================")
    logger.info(modelPath)
    logger.info(time.time()-All_start_time)

if __name__ == "__main__":
    main()







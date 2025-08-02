import os
import random
import sys

from Dataset import Dataset
from scipy.sparse import dok_matrix

# sys.path.append(os.path.join('/data/chenhai-fwxz/DeepCF-PyTorch/ncf-pytorch-master'))
import time
import argparse
import numpy as np
from matplotlib.pyplot import MultipleLocator
import matplotlib.pyplot as plt
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
# from tensorboardX import SummaryWriter
from utils import *
from utils_awp import AdvWeightPerturb1
import data_util
import evaluate
import evaluate2
# import GMF
# import MLP
from train_NeuMF import NeuMF

# def update_seed():
#     seed = torch.rand(9)  # 生成六个介于0到1之间的随机数
#     zs = [
#         1 if x > 0.5 else 0  # 如果x大于0.5，返回1；否则返回0
#         for x in seed
#     ]
#     return zs


def update_seed():
    T = random.uniform(0, 1)
    if T>0.5:
        return [1, 1, 1, 1, 1, 1, 1, 1, 1]
    else:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0] 

# def update_seed():
#     seed = torch.rand(9)  # 生成六个介于0到1之间的随机数
#     zs = [
#         1 if x > 0.5 else 0  # 如果x大于0.5，返回1；否则返回0
#         for x in seed
#     ]
#     return zs


parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate")
parser.add_argument("--batch_size", type=int, default=100, help="batch size for training")
parser.add_argument("--epochs", type=int, default=30, help="training epoches")
parser.add_argument("--top_k", type=int, default=10, help="compute metrics@top_k")
parser.add_argument("--embedding_dim_GMF", type=int, default=32, help="dimension of embedding in GMF submodel")
parser.add_argument("--embedding_dim_MLP", type=int, default=32, help="dimension of embedding in MLP submodel")
parser.add_argument("--hidden_layer_MLP", type=list, default=[64,32,16,8], help="hidden layers in MLP")
parser.add_argument("--use_pretrained", action="store_true",default=False, help="use pretrained model to initialize weights")
parser.add_argument("--num_ng", type=int, default=4, help="sample negative items for training")
parser.add_argument("--test_num_ng", type=int, default=99, help="sample part of negative items for testing")
parser.add_argument("--data_set", type=str, default="lastfm", help="data set. 'ml-1m' or 'pinterest-20'")
parser.add_argument('--awp-gamma', default=0.008, type=float)   # 0.001,0.005
parser.add_argument("--data_path", type=str, default="Data/")
parser.add_argument("--model_path", type=str, default="/results/model")
parser.add_argument("--out", default=False, help="save model or not")
# parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
parser.add_argument("--enable_lat", nargs="?", default=True)
# parser.add_argument("--epsilon", nargs="?", default=0.5)
# parser.add_argument("--alpha", nargs="?", default=1)
# parser.add_argument("--pro_num", nargs="?", default=1, choices=[1, 25], help="1 for fgsm and 10 for bim/pgd")
parser.add_argument("--decay_factor", nargs="?", default=1.0)
parser.add_argument("--layerlist", nargs="?", default="all")
parser.add_argument("--adv", nargs="?", default=True)
parser.add_argument("--adv_reg", nargs="?", default=1)
parser.add_argument("--reg", nargs="?", default=1e-3)
parser.add_argument("--adv_type", nargs="?", default="fgsm", choices=['fgsm', 'bim', 'pgd','mim'])
# parser.add_argument("--norm", nargs="?", default="linf", choices=['linf', 'l2'])
args = parser.parse_args()

def show_metric(epoch, ys,ys1, ys2,ys3, title1, title2, adv):
    """plot curlve for HR and NDCG
    Args:
        ys: hr or ndcg list
        title: HR@k or NDCG@k
    """
    plt.subplot(121)
    x = np.array(range(1, epoch+1))
    names=x
    y1 = ys
    y2 = ys2
    plt.plot(x, y1, 'b',label="NeuMF_MLP")
    plt.plot(x, y2, 'r',label="RAT-NeuMF_MLP")

    plt.xticks(x, names, rotation=1)
    plt.xlabel("Epoch", x = 0.5, y = -0.2, fontsize=14)
    plt.ylabel(title1,fontsize=14)
    x_major_locator=MultipleLocator(20)
    y_major_locator=MultipleLocator(0.2)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    #plt.yticks(np.arange(50, 90, 40))
    plt.xlim(-0.2,epoch)
    #把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    plt.ylim(-0.2,1)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2)

    plt.subplot(122)
    x = np.array(range(1, epoch+1))
    names=x
    y2 = ys1
    y3 = ys3
    plt.plot(x, y2, 'g',label="NeuMF")
    plt.plot(x, y3, 'r',label="RAT-NeuMF")

    plt.xticks(x, names, rotation=1)
    plt.xlabel("Epoch", x = 0.5, y = -0.2, fontsize=14)
    plt.ylabel(title2,fontsize=14)
    x_major_locator=MultipleLocator(20)
    y_major_locator=MultipleLocator(0.2)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    #plt.yticks(np.arange(50, 90, 40))
    plt.xlim(-0.2,epoch)
    #把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    plt.ylim(-0.2,1)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2)
    plt.subplots_adjust(left=0.15, bottom=0.1, right=0.95, top=0.8,wspace=0.4, hspace=0.2)
    plt.show()
    # plt.savefig("/data/chenhai-fwxz/DeepCF-PyTorch/result/%s_%s_%s.jpg" %(adv,title1, title2))



if __name__=="__main__":
    data_file = os.path.join(args.data_path, args.data_set)
    HR10=[]
    NDCG10=[]
    advHR10=[]
    advNDCG10=[]
    train_data, test_data, user_num, item_num, train_mat = data_util.load_all(data_file)

    
    dataset = Dataset(args.data_path + args.data_set)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    arr = np.array(test_data)
    max_value_second_column1 = np.max(arr[:, 1])
    arr = np.array(testNegatives)
    max_value_second_column2 = np.max(arr)
    n_value = max(max_value_second_column1, max_value_second_column2)
    if item_num <= n_value:
        item_num = n_value+1
        extended_sparse_matrix = dok_matrix((user_num, item_num), dtype=np.float32)

        for (row, col), value in train.items():
            extended_sparse_matrix[row, col] = value
            train = extended_sparse_matrix
    
    train_dataset = data_util.NCFData(train_data, item_num, train_mat, args.num_ng, True)
    test_dataset = data_util.NCFData(test_data, item_num, train_mat, 0, False)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_loader = data.DataLoader(test_dataset, batch_size=args.test_num_ng + 1, shuffle=False, num_workers=0, drop_last=True)

    GMF_model = None
    MLP_model = None
    model1 = NeuMF(user_num, item_num, args.embedding_dim_GMF, args.embedding_dim_MLP, args.hidden_layer_MLP, args.dropout, device, GMF_model,MLP_model)
    proxy_adv = NeuMF(user_num, item_num, args.embedding_dim_GMF, args.embedding_dim_MLP, args.hidden_layer_MLP, args.dropout, device, GMF_model,MLP_model)
    model1.to(device)
    proxy_adv.to(device)
    # RAWP_model2 = NeuMF(user_num, item_num, args.embedding_dim_MLP, args.hidden_layer_MLP, args.dropout,args.epsilon, args.pro_num,\
    #             args.enable_lat, args.layerlist, args.adv_type, args.adv_reg, args.norm,args.decay_factor, device, MLP_model)
    # model2.to(device)
    loss_function = nn.BCEWithLogitsLoss()

    if args.use_pretrained:
        optimizer1 = optim.SGD(model1.parameters(), lr=args.lr)
        proxy_opt = torch.optim.Adam(proxy_adv.parameters(), lr=args.lr_prox)
        # optimizer2 = optim.SGD(model2.parameters(), lr=args.lr)
    else:
        optimizer1 = optim.Adam(model1.parameters(), lr=args.lr)
        proxy_opt = torch.optim.Adam(proxy_adv.parameters(), lr=0.01)
        # optimizer2 = optim.Adam(model2.parameters(), lr=args.lr)

    awp_adversary = AdvWeightPerturb1(model=model1, proxy=proxy_adv, proxy_optim=proxy_opt, gamma=args.awp_gamma)
    ########################### TRAINING #####################################
    # HR2, NDCG2 = evaluate2.metrics(model2, test_loader, args.top_k, device)
    # print(f"Init model2: HR10={np.mean(HR2):.4f}, NDCG10={np.mean(NDCG2):.4f}")
    count1, best_hr1, count2, best_hr2 = 0, 0, 0, 0
    model1.train()
    # model2.train()
    # if args.enable_lat:
    #     model2.grad_init()
    for epoch1 in range(args.epochs):
          # Enable dropout (if have).
        start_time = time.time()
        train_loader.dataset.ng_sample()
        
        if epoch1>=0:

                # T = random.uniform(0.8, 1.2)
                # args.awp_gamma_temp = args.awp_gamma*T
                args.awp_gamma_temp = args.awp_gamma
                awp_adversary = AdvWeightPerturb1(model=model1, proxy=proxy_adv, proxy_optim=proxy_opt, gamma=args.awp_gamma_temp)
        
        seeds = update_seed()
        print(seeds,args.awp_gamma_temp, 1)

        for user, item, label in train_loader:
            user = user.to(device)
            item = item.to(device)
            label = label.float().to(device)

            awp = awp_adversary.calc_awp(inputs_adv_u=user, inputs_adv_i=item,inputs_adv_lab=label,
                                             attack_method=args.adv_type,pro_num=1,)
            awp_adversary.perturb(awp,seeds)
            
            optimizer1.zero_grad()
            prediction1 = model1(user, item)
            loss1 = loss_function(prediction1, label)
            loss1.backward()
            optimizer1.step()
            # writer.add_scalar('data/loss', loss.item(), count)
            count1 += 1
            awp_adversary.restore(awp,seeds)
        HR1, NDCG1 = evaluate.metrics(model1, test_loader, args.top_k, device)
        HR10.append(np.mean(HR1))
        NDCG10.append(np.mean(NDCG1))
        elapsed_time = time.time() - start_time
        print("NeuMF: The time elapse of epoch {:04d}".format(epoch1) + " is: " +
                time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        print("HR: {:.4f}\tNDCG: {:.4f}".format(np.mean(HR1), np.mean(NDCG1)))
        modelPath = f"pretrained/{args.data_set}_NeuMF_RAWP.pth"
        if HR1 > best_hr1:
            best_hr1, best_ndcg1, best_epoch1 = HR1, NDCG1, epoch1
            os.makedirs("pretrained", exist_ok=True)
            torch.save(model1.state_dict(), modelPath)
            print("save:=================================")
            print(modelPath)


    print("End model1. Best epoch {:04d}: HR = {:.4f}, NDCG = {:.4f}".format(best_epoch1, best_hr1, best_ndcg1))
    # print("End model2. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(best_epoch2, best_hr2, best_ndcg2))
    
    
    


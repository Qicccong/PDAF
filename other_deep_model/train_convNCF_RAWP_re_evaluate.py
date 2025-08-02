import load_dataset
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
from torch.autograd import Variable
import torch.multiprocessing as mp
import math
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from Dataset import Dataset
import evaluate1
import time
import random
import os
from scipy.sparse import dok_matrix

from utils_awp_ConNCF import AdvWeightPerturb1
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def update_seed():
    # seed = torch.rand(9)  # 生成六个介于0到1之间的随机数
    # zs = [
    #     1 if x > 0.5 else 0  # 如果x大于0.5，返回1；否则返回0
    #     for x in seed
    # ]
    # return zs
    seed = torch.rand(10)  # 生成六个介于0到1之间的随机数
    zs = [
        1 if x > 0.5 else 0  # 如果x大于0.5，返回1；否则返回0
        for x in seed
    ]
    return zs
    # T = random.uniform(0, 1)
    # if T>=0.5:
    #     return [1, 1, 1, 1, 1, 1, 1, 1, 1]
    # else:
    #     return [0, 0, 0, 0, 0, 0, 0, 0, 0]  

class ConvNCF(nn.Module):

    def __init__(self, user_count, item_count,device):
        super(ConvNCF, self).__init__()

        # some variables
        self.device = device
        self.user_count = user_count
        self.item_count = item_count
        # self.item_count = 12929 # AMusic

        # embedding setting
        self.embedding_size = 64

        self.P = nn.Embedding(self.user_count, self.embedding_size).to(self.device)
        self.Q = nn.Embedding(self.item_count, self.embedding_size).to(self.device)

        # cnn setting
        self.channel_size = 32
        self.kernel_size = 2
        self.strides = 2
        self.cnn = nn.Sequential(
            # batch_size * 1 * 64 * 64
            nn.Conv2d(1, self.channel_size, self.kernel_size, stride=self.strides),
            nn.ReLU(),
            # batch_size * 32 * 32 * 32
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            nn.ReLU(),
            # batch_size * 32 * 16 * 16
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            nn.ReLU(),
            # batch_size * 32 * 8 * 8
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            nn.ReLU(),
            # batch_size * 32 * 4 * 4
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            nn.ReLU(),
            # batch_size * 32 * 2 * 2
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            nn.ReLU(),
            # batch_size * 32 * 1 * 1
        )
        # self.cnn = nn.Sequential(
        #     # Input: batch × 1 × 64 × 64
        #     nn.Conv2d(1, 16, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),  # 输出: 16 × 32 × 32
            
        #     nn.Conv2d(16, 32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),  # 输出: 32 × 16 × 16
            
        #     nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 关键修改：输出通道改为64
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.AdaptiveAvgPool2d((1, 1))  # 输出: 64 × 1 × 1
        # )

        # self.fc = nn.Linear(32, 1)
        self.fc = nn.Sequential(
            nn.Linear(32, 16),  # 输入维度改为64
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 1)
        )
                

    def forward(self, user_ids, item_ids, is_pretrain):

        # convert float to int
        user_ids = list(map(int, user_ids))
        item_ids = list(map(int, item_ids))

        user_embeddings = self.P(torch.tensor(user_ids).to(self.device))
        item_embeddings = self.Q(torch.tensor(item_ids).to(self.device))

        if is_pretrain:
            # inner product
            prediction = torch.sum(torch.mul(user_embeddings, item_embeddings), dim=1)
        else:
            
            interaction_map = torch.bmm(user_embeddings.unsqueeze(2), item_embeddings.unsqueeze(1)).to(self.device)
            interaction_map = interaction_map.view((-1, 1, self.embedding_size, self.embedding_size)).to(self.device)

            # cnn
            feature_map = self.cnn(interaction_map).to(self.device)  # output: batch_size * 32 * 1 * 1
            feature_vec = feature_map.view((-1, 32)).to(self.device)

            # # CNN处理
            # feature_map = self.cnn(interaction_map).to(self.device)  # (batch, 64, 1, 1)
            # feature_vec = feature_map.view(-1, 64).to(self.device)  # (batch, 64)

            # fc
            prediction = self.fc(feature_vec).to(self.device)
            prediction = prediction.view((-1)).to(self.device)

        return prediction

class BPRLoss(nn.Module):

    def __init__(self):
        super(BPRLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, pos_preds, neg_preds):
        distance = pos_preds - neg_preds
        # loss = torch.sum(torch.log((1 + torch.exp(-distance))))
        sigmoid_distance = torch.sigmoid(distance)  # 先计算 sigmoid
        log_sigmoid = torch.log(sigmoid_distance)    # 再取对数
        loss = -torch.sum(log_sigmoid)
        
        return loss

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

def train(awp_gamma, model, train_data, testRatings, testNegatives):
    # lr = 0.5    # 原0.5
    lr = args.lr
    epoches = 100 # 原200
    bpr_epoches = 50
    batch_size = 100
    print(f"Batchsize: {batch_size}")
    losses = []
    accuracies = []
    bestHr10, bestNdcg10, bestepoch= 0, 0, -1
    model.train()
    print(f"oral_lr={lr},有学习策略,awp_gamma0.1")
    # optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=1e-2)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    proxy_opt = torch.optim.Adam(proxy_adv.parameters(), lr=0)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5, last_epoch=-1)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,momentum = 0.9, weight_decay=0.0005)
    # lr_decays = [int(epoches * 0.5), int(epoches * 0.75)]
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decays, gamma=0.1, last_epoch=-1)
    
    # train_loader = Data.DataLoader(dataset.train_group, batch_size=batch_size, shuffle=True, num_workers=4)
    # train_data = dataset.trainMatrix
    userInput, itemInput, labels = get_train_instances(train_data, 4)
    dst = BatchDataset(userInput, itemInput, labels)
    train_loader = Data.DataLoader(dst, batch_size=batch_size, shuffle=True, num_workers=4)
    
    bpr_loss = BPRLoss().cuda()
    # criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(epoches):    
        awp_adversary = AdvWeightPerturb1(model=model, proxy=proxy_adv, proxy_optim=proxy_opt, gamma=awp_gamma)
        seeds = update_seed()
        
        total_loss = 0
        total_acc = 0
        try:
            with tqdm(train_loader, disable=True) as t:
                for batch_idx, (user_ids, pos_item_ids, neg_item_ids) in enumerate(t):
                    user_ids = Variable(user_ids.cuda())
                    pos_item_ids = Variable(pos_item_ids.cuda())
                    neg_item_ids = Variable(neg_item_ids.cuda())
                    
                    optimizer.zero_grad()
                    
                    if epoch < bpr_epoches:    # 原来100
                        # pretrain bpr
                        pos_preds = model(user_ids, pos_item_ids, True)
                        neg_preds = model(user_ids, neg_item_ids, True)                        
                    else:
                        if epoch >= bpr_epoches:
                            awp = awp_adversary.calc_awp(user_ids=user_ids, pos_item_ids=pos_item_ids,neg_item_ids=neg_item_ids,
                                                attack_method='fgsm',pro_num=1,)
                            awp_adversary.perturb(awp,seeds)
                        # train convncf                        
                        pos_preds = model(user_ids, pos_item_ids, False)
                        neg_preds = model(user_ids, neg_item_ids, False)
                        
                    loss = bpr_loss(pos_preds, neg_preds)      
                    # loss = criterion(pos_preds, neg_preds)              
                    total_loss += loss.item()                    
                    loss.backward()
                    optimizer.step()
                    
                    if epoch >= bpr_epoches:
                        awp_adversary.restore(awp,seeds)
                    
                    accuracy = float(((pos_preds - neg_preds) > 0).sum()) / float(len(pos_preds))
                    total_acc += accuracy   
        except KeyboardInterrupt:
            t.close()
            raise
        
        losses.append(total_loss / (batch_idx + 1))
        accuracies.append(total_acc / (batch_idx + 1))
        print('RAWP_reevaluate_Epoch:', epoch, 'Train loss:', losses[-1], 'Train acc:', accuracies[-1])  
        if epoch % 5 == 0:
            t1 = time.time()
            # hr10, ndcg10 = evaluate()
            hits10, ndcgs10, maps10, mrrs10, hits20, ndcgs20,maps20, mrrs20,hits50, ndcgs50,maps50, mrrs50 = evaluate1.evaluate_model(model, testRatings, testNegatives, K=10, K1=20, K2=50,K3=100, num_thread=1, device=device)
            hr10, ndcg10,map10,mrr10, hr20, ndcg20,map20,mrr20,hr50, ndcg50,map50, mrr50 = np.array(hits10).mean(), np.array(ndcgs10).mean(),np.array(maps10).mean(), np.array(mrrs10).mean(),\
                np.array(hits20).mean(), np.array(ndcgs20).mean(), np.array(maps20).mean(), np.array(mrrs20).mean(), np.array(hits50).mean(), np.array(ndcgs50).mean(), np.array(maps50).mean(), np.array(mrrs50).mean()
            
            print(f"init: HR10={hr10:.4f}, NDCG10={ndcg10:.4f}")
            if hr10 > bestHr10:
                bestHr10, bestNdcg10, bestepoch = hr10, ndcg10, epoch
                modelPath = f"pretrained/{args.dataset}_lr-{args.lr}_g-{args.awp_gamma}_ConNCF_RAWP.pth"

                os.makedirs("pretrained", exist_ok=True)
                torch.save(model.state_dict(), modelPath)
                print("save:=================================")
                print(modelPath)   

        scheduler.step()
              

def evaluate(): # 5/11
    model.eval()
    hr5, hr10, hr20 = [], [], []
    ndcg5, ndcg10, ndcg20 = [], [], []

    user_count = len(dataset.testNegatives)
    try:
        with tqdm(range(user_count), disable=True) as t:
            for u in t:
                # 获取用户u的正样本ID
                pos_item = dataset.testRatings[u][1]
                # 生成候选列表：负样本 + 正样本
                item_candidates = dataset.testNegatives[u] + [pos_item]
                item_ids = torch.tensor(item_candidates).to(device)
                user_ids = torch.full((len(item_candidates),), u, dtype=torch.long).to(device)
                
                # 模型预测
                predictions = model(user_ids, item_ids, False)
                topv, topi = torch.topk(predictions, 20, dim=0)
                # hr, ndcg = scoreK(topi, 5)
                # hr5.append(hr)
                # ndcg5.append(ndcg)
                hr, ndcg = scoreK(topi, 10)
                hr10.append(hr)
                ndcg10.append(ndcg)
                # hr, ndcg = scoreK(topi, 20)
                # hr20.append(hr)
                # ndcg20.append(ndcg)                
        print('HR@10:', sum(hr10) / len(hr10))
        print('NDCG@10:', sum(ndcg10) / len(ndcg10))
    except KeyboardInterrupt:
        t.close()
        raise

    return sum(hr10) / len(hr10), sum(ndcg10) / len(ndcg10)

def scoreK(topi, k):
    hr = 1.0 if 999 in topi[0:k] else 0.0
    if hr:
        ndcg = math.log(2) / math.log(topi.tolist().index(999) + 2)
    else:
        ndcg = 0
    # auc = 1 - (position * 1. / negs)
    return hr, ndcg#, auc


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


def parse_args():
    parser = argparse.ArgumentParser(description="Run ConvNCF_RAWP.")
    parser.add_argument("--enable_lat", nargs="?", default=True)
    # parser.add_argument("--epsilon", nargs="?", default=0.5)
    parser.add_argument("--alpha", nargs="?", default=1)
    parser.add_argument("--pro_num", nargs="?", default=25, choices=[1, 25], help="1 for fgsm and 10 for bim/pgd")
    parser.add_argument("--decay_factor", nargs="?", default=1.0)
    parser.add_argument("--layerlist", nargs="?", default="all")
    parser.add_argument("--adv", nargs="?", default=True)
    parser.add_argument("--adv_reg", nargs="?", default=1)
    parser.add_argument("--adv_type", nargs="?", default="fgsm", choices=['fgsm', 'bim', 'pgd','mim'])
    parser.add_argument("--norm", nargs="?", default="linf", choices=['linf', 'l2'])
    parser.add_argument("--data_path", nargs="?", default="Data/",
                        help="Input data path.")
    parser.add_argument("--dataset", nargs="?", default="lastfm",   # yelp yelp  AToy yelp
                        help="Choose a dataset.")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learn rate")
    parser.add_argument('--awp-gamma', default=0.008, type=float)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("格式化时间:", formatted_time)
    print(f"mine_lr={args.lr}, dataset:{args.dataset}")
    awp_gamma = args.awp_gamma
    
    print('Data loading...')
    print('Data loaded')

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

    print('=' * 50)
    print('Model initializing...')

    model = ConvNCF(dataset.num_users, dataset.num_items, device=device).to(device)
    proxy_adv = ConvNCF(dataset.num_users, dataset.num_items, device).to(device)
    proxy_adv.load_state_dict(model.state_dict())   # 初始参数一致 5/11
    # proxy_opt = torch.optim.Adam(proxy_adv.parameters(), lr=0)
    proxy_opt = optim.Adagrad(proxy_adv.parameters(), lr=args.lr, weight_decay=1e-2)
    print('Model initialized')
    print('=' * 50)

    print('Model training...')
    train(awp_gamma, model, train_data, testRatings, testNegatives)
    print('Model trained')

    print('=' * 50)
    print('Model evaluating...')
    evaluate()
    print('Model evaluated')
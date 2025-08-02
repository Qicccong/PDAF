import load_dataset
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
from torch.autograd import Variable
import torch.multiprocessing as mp
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from utils_awp_ConNCF import AdvWeightPerturb1
import os
import argparse
import time
import evaluate1

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def show_curve(ys, title):
    """plot curlve for Loss and Accuacy
    Args:
        ys: loss or acc list
        title: Loss or Accuracy
    """
    x = np.array(range(len(ys)))
    y = np.array(ys)
    plt.plot(x, y, c='b')
    plt.axis()
    plt.title('{} Curve:'.format(title))
    plt.xlabel('Epoch')
    plt.ylabel('{} Value'.format(title))
    plt.show()

def show_metric(ys, title):
    """plot curlve for HR and NDCG
    Args:
        ys: hr or ndcg list
        title: HR@k or NDCG@k
    """
    x = np.array(range(len(ys)))
    y = np.array(ys)
    plt.plot(x, y, c='b')
    plt.axis()
    plt.title('{} Curve:'.format(title))
    plt.xlabel('User')
    plt.ylabel('{} Value'.format(title))
    plt.show()

class ConvNCF(nn.Module):

    def __init__(self, user_count, item_count):
        super(ConvNCF, self).__init__()

        # some variables
        self.user_count = user_count
        self.item_count = item_count
        # self.item_count = 12929

        # embedding setting
        self.embedding_size = 64

        self.P = nn.Embedding(self.user_count, self.embedding_size).cuda()
        self.Q = nn.Embedding(self.item_count, self.embedding_size).cuda()

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

        # fully-connected layer, used to predict
        self.fc = nn.Linear(32, 1)
        
        # dropout
#         self.drop_prop = 0.5
#         self.dropout = nn.Dropout(drop_prop)

    def forward(self, user_ids, item_ids, is_pretrain):

        # convert float to int
        user_ids = list(map(int, user_ids))
        item_ids = list(map(int, item_ids))

        user_embeddings = self.P(torch.tensor(user_ids).cuda())
        item_embeddings = self.Q(torch.tensor(item_ids).cuda())
        
        if is_pretrain:
            # inner product
            prediction = torch.sum(torch.mul(user_embeddings, item_embeddings), dim=1)
        else:
            # outer product
            # interaction_map = torch.ger(user_embeddings, item_embeddings) # ger is 1d
            interaction_map = torch.bmm(user_embeddings.unsqueeze(2), item_embeddings.unsqueeze(1))
            interaction_map = interaction_map.view((-1, 1, self.embedding_size, self.embedding_size))

            # cnn
            feature_map = self.cnn(interaction_map)  # output: batch_size * 32 * 1 * 1
            feature_vec = feature_map.view((-1, 32))

            # fc
            prediction = self.fc(feature_vec)
            prediction = prediction.view((-1))

        return prediction


class BPRLoss(nn.Module):

    def __init__(self):
        super(BPRLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, pos_preds, neg_preds):
        distance = pos_preds - neg_preds
        loss = torch.sum(torch.log((1 + torch.exp(-distance))))

        return loss


def train(awp_gamma):
    lr = args.lr
    epoches = 100 # 原200
    bpr_epoches = 20
    batch_size = 100
    print(f"Batchsize: {batch_size}")
    losses = []
    accuracies = []

    model.train()
    print("oral_lr=0.5,有学习策略,gamma0.001")
    bestHr10, bestNdcg10, bestepoch= 0,0, -1
    optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1, last_epoch=-1)

    train_loader = Data.DataLoader(yelp.train_group, batch_size=batch_size, shuffle=True, num_workers=4)
    
    bpr_loss = BPRLoss().cuda()

    for epoch in range(epoches):    
#         yelp.train_group = yelp.resample_train_group()
#         train_loader = Data.DataLoader(yelp.train_group, batch_size=batch_size, shuffle=True, num_workers=4)

        awp_adversary = AdvWeightPerturb1(model=model, proxy=proxy_adv, proxy_optim=proxy_opt, gamma=awp_gamma)
        
        seed = torch.rand(9)  # 生成六个介于0到1之间的随机数
        seeds = [
            1 if x > 0.5 else 0  # 如果x大于0.5，返回1；否则返回0
            for x in seed
        ]

        total_loss = 0
        total_acc = 0
        try:
            with tqdm(train_loader, disable=True) as t:
                for batch_idx, train_data in enumerate(t):                    
        #             train_data = Variable(train_data.cuda())
                    user_ids = Variable(train_data[:, 0].cuda())
                    pos_item_ids = Variable(train_data[:, 1].cuda())
                    neg_item_ids = Variable(train_data[:, 2].cuda())
                    
                    optimizer.zero_grad()
                    
#                     # pretrain bpr
#                     pos_preds = model(user_ids, pos_item_ids, True)
#                     neg_preds = model(user_ids, neg_item_ids, True) 

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
        print('RAWP_Epoch:', epoch, 'Train loss:', losses[-1], 'Train acc:', accuracies[-1])  
        if epoch % 5 == 0:
            # if epoch % 5 == 0 :      
                # show_curve(losses, 'train loss')
                # show_curve(accuracies, 'train acc')
                # evaluate()
            t1 = time.time()
            hr10, ndcg10 = evaluate()
            # hits10, ndcgs10, maps10, mrrs10, hits20, ndcgs20,maps20, mrrs20,hits50, ndcgs50,maps50, mrrs50 = evaluate1.evaluate_model(model, testRatings, testNegatives, 10, 20, 50,100, 1, device)
            # hr10, ndcg10,map10,mrr10, hr20, ndcg20,map20,mrr20,hr50, ndcg50,map50, mrr50 = np.array(hits10).mean(), np.array(ndcgs10).mean(),np.array(maps10).mean(), np.array(mrrs10).mean(),\
            #     np.array(hits20).mean(), np.array(ndcgs20).mean(), np.array(maps20).mean(), np.array(mrrs20).mean(), np.array(hits50).mean(), np.array(ndcgs50).mean(), np.array(maps50).mean(), np.array(mrrs50).mean()
            print(f"init: HR10={hr10:.4f}, NDCG10={ndcg10:.4f}")
            if hr10 > bestHr10:
                bestHr10, bestNdcg10, bestepoch = hr10, ndcg10, epoch
                
                modelPath = f"pretrained/{args.dataset}_lr-{args.lr}_g-{args.awp_gamma}_ConNCF_RAWP.pth"

                os.makedirs("pretrained", exist_ok=True)
                torch.save(model.state_dict(), modelPath)
                print("save:=================================")
                print(modelPath)   
        # if epoch == 3 or epoch == 20:
        #     evaluate()
        scheduler.step()
              

def evaluate():
    model.eval()
    hr5 = []
    hr10 = []
    hr20 = []
    ndcg5 = []
    ndcg10 = []
    ndcg20 = []

    user_count = len(yelp.test_negative)
    try:
        with tqdm(range(user_count), disable=True) as t:
            for u in t:
                item_ids = torch.tensor(yelp.test_negative[u]).cuda()
                user_ids = torch.tensor([u] * len(item_ids)).cuda()
                predictions = model(user_ids, item_ids, False)
                topv, topi = torch.topk(predictions, 20, dim=0)
                hr, ndcg = scoreK(topi, 5)
                hr5.append(hr)
                ndcg5.append(ndcg)
                hr, ndcg = scoreK(topi, 10)
                hr10.append(hr)
                ndcg10.append(ndcg)
                hr, ndcg = scoreK(topi, 20)
                hr20.append(hr)
                ndcg20.append(ndcg)

            # show_metric(hr5, 'HR@5')
            # show_metric(ndcg5, 'NDCG@5')
            # show_metric(hr10, 'HR@10')
            # show_metric(ndcg10, 'NDCG@10')
            # show_metric(hr20, 'HR@20')
            # show_metric(ndcg20, 'NDCG@20')            
            # print('HR@5:', sum(hr5))
            # print(len(hr5))
            # print('NDCG@5:', sum(ndcg5) / len(ndcg5))
            print('HR@10:', sum(hr10) / len(hr10))
            print('NDCG@10:', sum(ndcg10) / len(ndcg10))
            # print('HR@20:', sum(hr20) / len(hr20))
            # print('NDCG@20:', sum(ndcg20) / len(ndcg20))
    except KeyboardInterrupt:
        t.close()
        raise

    return sum(hr10) / len(hr10), sum(ndcg10) / len(ndcg10)

# def scoreK(topi, k):
#     hr = 1.0 if 999 in topi[0:k, 0] else 0.0
#     if hr:
#         ndcg = math.log(2) / math.log(topi[:, 0].tolist().index(999) + 2)
#     else:
#         ndcg = 0
#     # auc = 1 - (position * 1. / negs)
#     return hr, ndcg#, auc

def scoreK(topi, k):
    hr = 1.0 if 999 in topi[0:k] else 0.0
    if hr:
        ndcg = math.log(2) / math.log(topi.tolist().index(999) + 2)
    else:
        ndcg = 0
    # auc = 1 - (position * 1. / negs)
    return hr, ndcg#, auc

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
    parser.add_argument('--awp-gamma', default=0.001, type=float)
    return parser.parse_args()

if __name__ == '__main__':
    # torch.set_num_threads(12)
    args = parse_args()
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("格式化时间:", formatted_time)
    print(f"mine_lr={args.lr}, dataset:{args.dataset}")
    awp_gamma = args.awp_gamma
    print('Data loading...')
    yelp = load_dataset.Load_Yelp(f'./Data/{args.dataset}.train.rating', f'./Data/{args.dataset}.test.rating', f'./Data/{args.dataset}.test.negative')
    print('Data loaded')

    print('=' * 50)
    print('Model initializing...')
    model = ConvNCF(int(max(yelp.train_group[:, 0])) + 1, int(max(yelp.train_group[:, 1])) + 1).cuda()
    proxy_adv = ConvNCF(int(max(yelp.train_group[:, 0])) + 1, int(max(yelp.train_group[:, 1])) + 1).cuda()
    
    proxy_opt = optim.Adagrad(proxy_adv.parameters(), lr=0.01, weight_decay=1e-2)
    # awp_adversary = AdvWeightPerturb1(model=model, proxy=proxy_adv, proxy_optim=proxy_opt, gamma=awp_gamma)
    print('Model initialized')

    print('=' * 50)

    print('Model training...')
    train(awp_gamma)
    # train(awp_gamma,model, testRatings, testNegatives)
    print('Model trained')

    print('=' * 50)

    print('Model evaluating...')
    evaluate()
    print('Model evaluated')
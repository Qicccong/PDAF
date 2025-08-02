import load_dataset
import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
from matplotlib.pyplot import MultipleLocator
from torch.autograd import Variable
import torch.multiprocessing as mp
import numpy as np
import math
from tqdm import tqdm
import time
import os
from Dataset import Dataset
import evaluate1
import matplotlib.pyplot as plt
# device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def parse_args():
    parser = argparse.ArgumentParser(description="Run ConvNCF.")
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
    parser.add_argument("--dataset", nargs="?", default="lastfm",   # lastfm yelp  AToy lastfm
                        help="Choose a dataset.")
    parser.add_argument("--lr", type=float, default=0.01,   # lastfm yelp  AToy lastfm
                        help="learn rate")
    return parser.parse_args()


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
    plt.plot(x, y1, 'b',label="ConvNCF")
    plt.plot(x, y2, 'r',label="RAT-ConvNCF")

    plt.xticks(x, names, rotation=1)
    plt.xlabel("Epoch", x = 0.5, y = -0.2, fontsize=14)
    plt.ylabel(title1,fontsize=14)
    x_major_locator=MultipleLocator(300)
    y_major_locator=MultipleLocator(0.03)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    #plt.yticks(np.arange(50, 90, 40))
    plt.xlim(-1,epoch)
    #把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    plt.ylim(0,0.3)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2)

    plt.subplot(122)
    x = np.array(range(1, epoch+1))
    names=x
    y2 = ys1
    y3 = ys3
    plt.plot(x, y2, 'g',label="ConvNCF")
    plt.plot(x, y3, 'r',label="RAT-ConvNCF")

    plt.xticks(x, names, rotation=1)
    plt.xlabel("Epoch", x = 0.5, y = -0.2, fontsize=14)
    plt.ylabel(title2,fontsize=14)
    x_major_locator=MultipleLocator(300)
    y_major_locator=MultipleLocator(0.03)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    #plt.yticks(np.arange(50, 90, 40))
    plt.xlim(-1,epoch)
    #把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    plt.ylim(0,0.3)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2)
    plt.subplots_adjust(left=0.15, bottom=0.1, right=0.95, top=0.8,wspace=0.4, hspace=0.2)
    plt.show()
    plt.savefig("/data/chenhai-fwxz/DeepCF-PyTorch/result/%s_%s_%s.jpg" %(adv,title1, title2))

class ConvNCF(nn.Module):

    def __init__(self, user_count, item_count,device):
        super(ConvNCF, self).__init__()

        # some variables
        self.device = device
        self.user_count = user_count
        self.item_count = item_count
        # self.item_count = 12929

        print(item_count)
        print("*"*50)
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

        self.fc = nn.Linear(32, 1)
        

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
        loss = torch.sum(torch.log((1 + torch.exp(-distance))))
        

        return loss


def cal_lp_norm(tensor,p,dim_count):
    tmp = tensor
    for i in range(1,dim_count):
        tmp = torch.norm(tmp,p=p,dim=i,keepdim=True) #torch.Size([100, 1])
    
    return torch.clamp_min(tmp, 1e-8)


def set_noise(epoch,adv_type):
    if 0<=epoch < 300:
        if adv_type=="fgsm":
            return 0.5,1
        else:
            return 0.5, 25
    elif 300<=epoch < 600:
        if adv_type=="fgsm":
            return 0.4,1
        else:
            return 0.4, 20
    elif 600<=epoch < 900:
        if adv_type=="fgsm":
            return 0.6,1
        else:
            return 0.6, 20
    elif 900<=epoch < 1200:
        if adv_type=="fgsm":
            return 0.3,1
        else:
            return 0.3, 15
    
    elif 1200<=epoch<= 1500:
        if adv_type=="fgsm":
            return 0.2,1
        else:
            return 0.2, 10

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

def train(model, testRatings, testNegatives,adversarial):
    # lr = 0.1
    lr = args.lr
    print(f"mine_lr={lr},dataset:{args.dataset}")
    epoches = 300  # 1500
    batch_size = 100  # 1000
    losses = []
    accuracies = []
    H10 = []
    NDCG10 = []
    bestHr10, bestNdcg10, bestepoch= 0,0, -1
    model.train()
    
    optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.05, last_epoch=-1)

    train_loader = Data.DataLoader(yelp.train_group, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    # train_data = dataset.trainMatrix
    # userInput, itemInput, labels = get_train_instances(train_data, 0)
    # dst = BatchDataset(userInput, itemInput, labels)
    # train_loader = Data.DataLoader(dst, batch_size=batch_size, shuffle=True, num_workers=0)
    
    bpr_loss = BPRLoss().to(device)

    for epoch in range(epoches):    

        total_loss = 0
        total_acc = 0
        try:
            with tqdm(train_loader, disable=True) as t:
                for batch_idx, train_data in enumerate(t):                    
        #             train_data = Variable(train_data.to(self.device))
                    user_ids = Variable(train_data[:, 0].to(device))
                    pos_item_ids = Variable(train_data[:, 1].to(device))
                    neg_item_ids = Variable(train_data[:, 2].to(device))
                    
                    optimizer.zero_grad()
                
                    if epoch < 100:     # 100
                        # pretrain bpr
                        pos_preds = model(user_ids, pos_item_ids, True)
                        neg_preds = model(user_ids, neg_item_ids, True)                        
                    else:
                        # train convncf                        
                        pos_preds = model(user_ids, pos_item_ids, False)
                        neg_preds = model(user_ids, neg_item_ids, False)
                        
                    loss = bpr_loss(pos_preds, neg_preds)                    
                    total_loss += loss.item()                    
                    loss.backward()
                    optimizer.step()
                    
                    accuracy = float(((pos_preds - neg_preds) > 0).sum()) / float(len(pos_preds))
                    total_acc += accuracy
        #             if batch_idx % 20 == 0:
        #                 print('Epoch:', epo.ch, 'Batch:', batch_idx, 'Train loss:', losses[-1], 'Train acc:', accuracies[-1])                                
        except KeyboardInterrupt:
            t.close()
            raise
        
        losses.append(total_loss / (batch_idx + 1))
        accuracies.append(total_acc / (batch_idx + 1))
        print('wo_gai_clean_Epoch:', epoch, 'Train loss:', losses[-1], 'Train acc:', accuracies[-1])  

        scheduler.step()
        # if epoch== 5 or epoch>= 18: # 98
        if epoch % 5 == 0 :
            t1 = time.time()
            # hr10, ndcg10 = evaluate(epoch,model)
            hits10, ndcgs10, maps10, mrrs10, hits20, ndcgs20,maps20, mrrs20,hits50, ndcgs50,maps50, mrrs50 = evaluate1.evaluate_model(model, testRatings, testNegatives, 10, 20, 50,100, 1, device)
            hr10, ndcg10,map10,mrr10, hr20, ndcg20,map20,mrr20,hr50, ndcg50,map50, mrr50 = np.array(hits10).mean(), np.array(ndcgs10).mean(),np.array(maps10).mean(), np.array(mrrs10).mean(),\
                np.array(hits20).mean(), np.array(ndcgs20).mean(), np.array(maps20).mean(), np.array(mrrs20).mean(), np.array(hits50).mean(), np.array(ndcgs50).mean(), np.array(maps50).mean(), np.array(mrrs50).mean()
            # print(hr20)
            # print(hr50)
            print(f"init: HR10={hr10:.4f}, NDCG10={ndcg10:.4f}, mrrs10={mrr10:.4f}, HR20={hr20:.4f}, NDCG20={ndcg20:.4f}, mrrs20={mrr20:.4f}, HR50={hr50:.4f}, NDCG50={ndcg50:.4f}, mrrs50={mrr50:.4f} [{time.time()-t1:.1f}s]")
            if hr10 > bestHr10:
                    bestHr10, bestNdcg10, bestepoch = hr10, ndcg10, epoch
            
                    modelPath = f"pretrained/{args.dataset}_lr-{args.lr}_ConNCF.pth"

                    os.makedirs("pretrained", exist_ok=True)
                    torch.save(model.state_dict(), modelPath)
                    print("save:=================================")
                    print(modelPath)               
    print(f"Best epoch {bestepoch+1}: HR10={bestHr10:.4f}, NDCG10={bestNdcg10:.4f}")
    return H10, NDCG10

def scoreK(topi, k,item):
    hr = 1.0 if 999 in topi[0:k] else 0.0
    if hr:
        ndcg = math.log(2) / math.log(topi.tolist().index(999) + 2)
    else:
        ndcg = 0
    # auc = 1 - (position * 1. / negs)
    return hr, ndcg#, auc


if __name__ == '__main__':
    local_time = time.localtime()
    print(time.strftime("%Y-%m-%d %H:%M:%S", local_time))

    # torch.set_num_threads(12)
    args = parse_args()
    print(f'Data {args.dataset} loading...')
    yelp = load_dataset.Load_Yelp(f'./Data/{args.dataset}.train.rating', f'./Data/{args.dataset}.test.rating', f'./Data/{args.dataset}.test.negative')
    print('Data loaded')
    dataset = Dataset(args.data_path + args.dataset)
    # train_data, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    _, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    print('=' * 50)

    print('Model initializing...')
    # 计算所有用户和物品的最大 ID（包括训练和测试）
    # max_user_id = max(int(max(yelp.train_group[:, 0])), int(np.max(testRatings[:, 0])))
    # max_item_id = max(int(max(yelp.train_group[:, 1])), int(np.max(testRatings[:, 1])))

    # model1 = ConvNCF(max_user_id + 1, max_item_id + 1, device).to(device)
    model1 = ConvNCF(int(max(yelp.train_group[:, 0])) + 1, int(max(yelp.train_group[:, 1])) + 1, device).to(device)
    # model1 = ConvNCF(dataset.num_users, dataset.num_items, device=device).to(device)
    # proxy_adv = ConvNCF(dataset.num_users, dataset.num_items, device).to(device)
    # proxy_adv.load_state_dict(model1.state_dict())   # 初始参数一致 5/11
    # proxy_opt = optim.Adagrad(proxy_adv.parameters(), lr=args.lr, weight_decay=1e-2)
    # model2 = RAT_ConvNCF(int(max(yelp.train_group[:, 0])) + 1, int(max(yelp.train_group[:, 1])) + 1, args.enable_lat, args.layerlist, args.adv_type, args.adv_reg, args.norm,args.decay_factor,args.epsilon, args.pro_num, device).to(device)
    print('Model initialized')

    print('=' * 50)

    print('Model training...')
    H10,ndcg10=train(model1, testRatings, testNegatives,False)
    # HR10, NDCG10=train2(model2)
    print('Model trained')
    # show_metric(1500, H10,ndcg10, 'HR@10', 'NDCG@10')
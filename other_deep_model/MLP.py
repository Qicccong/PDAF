import os
import time
import argparse
import numpy as np
from matplotlib.pyplot import MultipleLocator
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
device = 'cuda:6' if torch.cuda.is_available() else 'cpu'
# from tensorboardX import SummaryWriter

import data_util
import evaluate
import evaluate2

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.00001, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate")
parser.add_argument("--batch_size", type=int, default=100, help="batch size for training")
parser.add_argument("--epochs", type=int, default=50, help="training epoches")
parser.add_argument("--top_k", type=int, default=10, help="compute metrics@top_k")
parser.add_argument("--embedding_dim", type=int, default=128, help="dimension of embedding")
parser.add_argument("--hidden_layer", type=list, default=[256,128, 64, 32], help="dimension of each hidden layer")
parser.add_argument("--num_ng", type=int, default=4, help="sample negative items for training")
parser.add_argument("--test_num_ng", type=int, default=99, help="sample part of negative items for testing")
parser.add_argument("--data_set", type=str, default="lastfm", help="data set. 'ml-1m' or 'pinterest-20'")
parser.add_argument("--data_path", type=str, default="/data/chenhai-fwxz/DeepCF-PyTorch/Data")
# parser.add_argument("--model_path", type=str, default="/Users/JingboLiu/Desktop/ncf-pytorch/model")

parser.add_argument("--enable_lat", nargs="?", default=True)
parser.add_argument("--epsilon", nargs="?", default=0.5)
parser.add_argument("--alpha", nargs="?", default=1)
parser.add_argument("--pro_num", nargs="?", default=25, choices=[1, 25], help="1 for fgsm and 10 for bim/pgd")
parser.add_argument("--decay_factor", nargs="?", default=1.0)
parser.add_argument("--layerlist", nargs="?", default="all")
parser.add_argument("--adv", nargs="?", default=True)
parser.add_argument("--adv_reg", nargs="?", default=1)
parser.add_argument("--reg", nargs="?", default=1e-3)
parser.add_argument("--adv_type", nargs="?", default="mim", choices=['fgsm', 'bim', 'pgd','mim'])
parser.add_argument("--norm", nargs="?", default="linf", choices=['linf', 'l2'])
args = parser.parse_args()

def cal_lp_norm(tensor,p,dim_count):
    tmp = tensor
    for i in range(1,dim_count):
        tmp = torch.norm(tmp,p=p,dim=i,keepdim=True) #torch.Size([100, 1])
    
    
    return torch.clamp_min(tmp, 1e-8)



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
    x_major_locator=MultipleLocator(10)
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
    x_major_locator=MultipleLocator(10)
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
    plt.savefig("/data/chenhai-fwxz/DeepCF-PyTorch/result/lastfm_%s_%s_%s.jpg" %(adv,title1, title2))


class MLP(nn.Module):
    def __init__(self, user_num, item_num, embedding_dim, hidden_layer, dropout):
        super(MLP, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        embedding_dim: number of embedding dimensions;
        hidden_layer: dimension of each hidden layer (list type);
        dropout: dropout rate between fully connected layers.
        """
        self.dropout = dropout

        self.embed_user = nn.Embedding(user_num, embedding_dim)
        self.embed_item = nn.Embedding(item_num, embedding_dim)

        # MLP_modules = []
        # self.num_layers = len(hidden_layer)
        # for i in range(self.num_layers):
        #     MLP_modules.append(nn.Dropout(p=self.dropout))
        #     if i == 0:
        #         MLP_modules.append(nn.Linear(embedding_dim*2, hidden_layer[0]))
        #     else:
        #         MLP_modules.append(nn.Linear(hidden_layer[i-1], hidden_layer[i]))
        #     MLP_modules.append(nn.ReLU())
        # self.MLP_layers = nn.Sequential(*MLP_modules)

        # self.predict_layer = nn.Linear(hidden_layer[-1], 1)
        
        MLP_modules = []
        for l1, l2 in zip(hidden_layer[:-1], hidden_layer[1:]):
            MLP_modules.append(nn.Linear(l1, l2))
            MLP_modules.append(nn.ReLU(inplace=True))
        self.MLP_layers = nn.Sequential(*MLP_modules)

        # In the official implementation, 
        # the final module is initialized using Lecun normal method.
        # Here, the Kaiming normal initialization is adopted.
        self.predict_layer = nn.Linear(hidden_layer[-1], 1)
            
        
        

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')


        # Kaiming/Xavier initialization can not deal with non-zero bias terms
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, user, item):
        embed_user = self.embed_user(user)
        embed_item = self.embed_item(item)
        interaction = torch.cat((embed_user, embed_item), -1)
        output = self.MLP_layers(interaction)

        prediction = self.predict_layer(output)
        return prediction.view(-1)


class RAT_MLP(nn.Module):
    def __init__(self, user_num, item_num, embedding_dim_MLP, hidden_layer_MLP,
                 dropout, epsilon, pro_num, enable_lat, layerlist, adv_type, adv_reg,norm,decay_factor,device):
        super(RAT_MLP, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        embedding_dim_GMF: number of embedding dimensions in GMF submodel;
        embedding_dim_MLP: number of embedding dimensions in MLP submodel;
        hidden_layer_MLP: dimension of each hidden layer (list type) in MLP submodel;
        dropout: dropout rate between fully connected layers;
        GMF_model: pre-trained GMF weights;
		MLP_model: pre-trained MLP weights.
        """
        self.loss_function = nn.BCEWithLogitsLoss()
        self.device = device
        self.epsilon = epsilon
        self.adv_type = adv_type
        self.norm = norm
        self.adv_reg = adv_reg
        self.decay_factor = decay_factor
        self.pro_num = pro_num
        self.enable_lat = enable_lat
        self.seed1,_,_, = self.random()
        _,self.seed2,_ = self.random()
        _,_,self.seed3 = self.random()
        self.maxseed = max(self.seed1, self.seed2,self.seed3)
        self.y_list = ['y0_reg','y1_reg','y2_reg','y3_reg']
        self.enable_list = [0 for i in range(4)]
        if enable_lat and layerlist != "all":
            self.layerlist = [int(x) for x in layerlist.split(',')]
            self.layerlist_digit = [int(x) for x in layerlist.split(',')]
        else:
            self.layerlist = "all"
            self.layerlist_digit = list(range(0,self.maxseed+1))
        self.dropout = dropout

        self.embed_user_MLP = nn.Embedding(user_num, embedding_dim_MLP).to(self.device)
        self.embed_item_MLP = nn.Embedding(item_num, embedding_dim_MLP).to(self.device)


        MLP_modules = []
        self.reg_size_list = list()
        y_index = 1
        self.register_buffer(self.y_list[0], torch.zeros([100, 256]))
        for l1, l2 in zip(hidden_layer_MLP[:-1], hidden_layer_MLP[1:]):
            MLP_modules.append(nn.Linear(l1, l2))
            MLP_modules.append(nn.ReLU(inplace=True))
            self.register_buffer(self.y_list[y_index], torch.zeros([100, l2]))
            self.reg_size_list.append([100, l2])
            y_index+=1
        self.MLP_layers = nn.Sequential(*MLP_modules)

        self.predict_layer = nn.Linear(hidden_layer_MLP[-1], 1)
        
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')


        # Kaiming/Xavier initialization can not deal with non-zero bias terms
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()
        self.choose_layer()

    def forward(self, user, item, label):

        embed_user_MLP = self.embed_user_MLP(user).to(self.device)
        embed_item_MLP = self.embed_item_MLP(item).to(self.device)
        
        self.y0 = torch.cat((embed_user_MLP, embed_item_MLP), -1).to(self.device)
        
        self.y1 = self.MLP_layers[0](self.y0).to(self.device)
        self.y1 = self.MLP_layers[1](self.y1) 
        
        self.y2 = self.MLP_layers[2](self.y1).to(self.device)
        self.y2 = self.MLP_layers[3](self.y2)
        
        self.y3 = self.MLP_layers[4](self.y2).to(self.device)
        self.y3 = self.MLP_layers[5](self.y3)

        prediction = self.predict_layer(self.y3).view(-1)
        prediction = prediction.to(self.device)
        loss = self.loss_function(prediction, label.float())

        if self.enable_lat:
            
            self.y1 = self.MLP_layers[0](self.y0).to(self.device)  #torch.Size([100, 256])         
            if self.enable_lat and self.enable_list1:
                self.y1.retain_grad()
                self.y1_add = self.y1.add(self.y1_reg.data)
                if self.norm =="linf":
                        delta = torch.clamp(self.y1_add-self.y1, -self.epsilon, self.epsilon).to(self.device) 
                        self.y1_add = self.y1.add(delta).to(self.device) 
                if self.norm =="l2":
                    delta = self.y1_add-self.y1
                    normVal = torch.norm(delta.view(100, -1), 2, 1)
                    mask = normVal<=self.epsilon
                    scaling = self.epsilon/normVal
                    scaling[mask] = 1
                    delta = delta*scaling.view(100, 1)
                    self.y1_add = self.y1.add(delta).to(self.device) 
            else:
                self.y1_add = self.y1
            self.z1 = self.MLP_layers[1](self.y1_add).to(self.device) 
            
            self.y2 = self.MLP_layers[2](self.z1).to(self.device) 
            if self.enable_lat and self.enable_list2:
                self.y2.retain_grad()
                self.y2_add = self.y2.add(self.y2_reg.data)
                if self.norm =="linf":
                        delta = torch.clamp(self.y2_add-self.y2, -self.epsilon, self.epsilon).to(self.device) 
                        self.y2_add = self.y2.add(delta).to(self.device) 
                if self.norm =="l2":
                    delta = self.y2_add-self.y2
                    normVal = torch.norm(delta.view(100, -1), 2, 1)
                    mask = normVal<=self.epsilon
                    scaling = self.epsilon/normVal
                    scaling[mask] = 1
                    delta = delta*scaling.view(100, 1)
                    self.y2_add = self.y2.add(delta).to(self.device) 
            else:
                self.y2_add = self.y2
            self.z2 = self.MLP_layers[3](self.y2_add).to(self.device)  #torch.Size([100, 128])

            self.y3 = self.MLP_layers[4](self.z2).to(self.device)   
            if self.enable_lat and self.enable_list3:
                self.y3.retain_grad()
                self.y3_add = self.y3.add(self.y3_reg.data)
                if self.norm =="linf":
                        delta = torch.clamp(self.y3_add-self.y3, -self.epsilon, self.epsilon).to(self.device)  
                        self.y3_add = self.y3.add(delta).to(self.device) 
                if self.norm =="l2":
                    delta = self.y3_add-self.y3
                    normVal = torch.norm(delta.view(100, -1), 2, 1)
                    mask = normVal<=self.epsilon
                    scaling = self.epsilon/normVal
                    scaling[mask] = 1
                    delta = delta*scaling.view(100, 1)
                    self.y3_add = self.y3.add(delta).to(self.device) 
            else:
                self.y3_add = self.y3
            self.z3 = self.MLP_layers[5](self.y3_add).to(self.device) 
            # concat = torch.cat((output_GMF, self.z3
            prediction = self.predict_layer(self.z3).view(-1)
            prediction = prediction.to(self.device)
            loss_adv = self.loss_function(prediction, label.float())
            loss = loss + self.adv_reg * loss_adv 

        return prediction, loss

    def grad_init(self):
        if self.adv_type=="naive":
            for i in range(0,self.maxseed+1): 
                exec('self.y{}_reg.data = torch.randn(self.y{})'.format(i,i))
        elif self.adv_type=="fgsm" or self.adv_type=="bim" or self.adv_type=="mim":
            for i in range(0,self.maxseed+1): 
                exec('self.y{}_reg.data = torch.zeros_like(self.y{}).detach()'.format(i,i))
        elif self.adv_type=="pgd":
            for i in range(0,self.maxseed+1):
                if self.norm=="linf": 
                    exec('self.y{}_reg.data = torch.empty_like(self.y{}).uniform_(-self.epsilon, self.epsilon)'.format(i,i,i))
                elif self.norm=="l2":
                    exec('delta{} = torch.empty_like(self.y{}).uniform_(-self.epsilon, self.epsilon)'.format(i,i))
                    exec('normval{} = torch.norm(delta{}.view(100, -1), 2, 1)'.format(i,i))
                    exec('mask{} = normVal{}<=self.epsilon'.format(i,i))
                    exec('scaling{} = self.epsilon/normVal{}'.format(i,i))
                    exec('scaling{}[mask{}] = 1'.format(i,i))
                    exec('delta{} = delta{}*scaling{}.view(100, 1)'.format(i,i,i))
                    exec('self.y{}_reg.data = delta{}'.format(i,i))


    def choose_layer(self):
        if self.enable_lat == False:
            return
        if self.layerlist == 'all':
            self.enable_list1 = list(range(0, self.seed1+1))
            self.enable_list2 = list(range(0, self.seed2+1))
            self.enable_list3 = list(range(0, self.seed3+1))
  
        else:
            for i in self.layerlist_digit:
                self.enable_list[i] = 1

    def save_grad(self):
        if self.enable_lat:
            if 1 in self.enable_list1:
                if self.norm=="linf" and self.adv_type!="mim":
                    self.y1_reg.data = (self.epsilon / self.pro_num) * (self.y1.grad.sign())
                elif self.norm=="l2" and self.adv_type!="mim":
                    self.y1_reg.data = (self.epsilon / self.pro_num) * (self.y1.grad / cal_lp_norm(self.y1.grad,p=2,dim_count=len(self.y1_add.grad.size())))
                elif self.adv_type=="mim":
                    momentum1 = torch.zeros_like(self.y1).detach()
                    grad1 = self.y1.grad
                    grad_norm1 = torch.norm(nn.Flatten()(grad1), p=1, dim=1)
                    grad1 = grad1 / grad_norm1.view([-1]+[1]*(len(grad1.shape)-1))
                    grad1 = grad1 + self.decay_factor * momentum1
                    momentum1 = grad1
                    if self.norm=="linf":
                        self.y1_reg.data = (self.epsilon / self.pro_num) * (grad1.sign())

                    elif self.norm=="l2":
                        self.y1_reg.data = (self.epsilon / self.pro_num) * (grad1 / cal_lp_norm(grad1,p=2,dim_count=len(self.y1.grad.size())))
            
            if 2 in self.enable_list2:
                if self.norm=="linf" and self.adv_type!="mim":
                    self.y2_reg.data = (self.epsilon / self.pro_num) * (self.y2.grad.sign())
                elif self.norm=="l2" and self.adv_type!="mim":
                    self.y2_reg.data = (self.epsilon / self.pro_num) * (self.y2.grad / cal_lp_norm(self.y2.grad,p=2,dim_count=len(self.y2_add.grad.size())))
                elif self.adv_type=="mim":
                    momentum2 = torch.zeros_like(self.y2).detach()
                    grad2 = self.y2.grad
                    grad_norm2 = torch.norm(nn.Flatten()(grad2), p=1, dim=1)
                    grad2 = grad2 / grad_norm2.view([-1]+[1]*(len(grad2.shape)-1))
                    grad2 = grad2 + self.decay_factor * momentum2
                    momentum2 = grad2
                    if self.norm=="linf":
                        self.y2_reg.data = (self.epsilon / self.pro_num) * (grad2.sign())

                    elif self.norm=="l2":
                        self.y2_reg.data = (self.epsilon / self.pro_num) * (grad2 / cal_lp_norm(grad2,p=2,dim_count=len(self.y2.grad.size())))

            if 3 in self.enable_list3:
                if self.norm=="linf" and self.adv_type!="mim":
                    self.y3_reg.data = (self.epsilon / self.pro_num) * (self.y3.grad.sign())
                elif self.norm=="l2" and self.adv_type!="mim":
                    self.y3_reg.data = (self.epsilon / self.pro_num) * (self.y3.grad / cal_lp_norm(self.y3.grad,p=2,dim_count=len(self.y3_add.grad.size())))
                elif self.adv_type=="mim":
                    momentum3 = torch.zeros_like(self.y3).detach()
                    grad3 = self.y3.grad
                    grad_norm3 = torch.norm(nn.Flatten()(grad3), p=1, dim=1)
                    grad3 = grad3 / grad_norm3.view([-1]+[1]*(len(grad3.shape)-1))
                    grad3 = grad3 + self.decay_factor * momentum3
                    momentum3 = grad3
                    if self.norm=="linf":
                        self.y3_reg.data = (self.epsilon / self.pro_num) * (grad3.sign())

                    elif self.norm=="l2":
                        self.y3_reg.data = (self.epsilon / self.pro_num) * (grad3 / cal_lp_norm(grad3,p=2,dim_count=len(self.y3.grad.size())))
 
    def update_seed(self):
        self.seed1,self.seed2,self.seed3 = self.random()

    def random(self):
        seed = torch.rand(3)*0.4
        zs1= int(torch.clamp(seed[0]*10, min=0, max=3))
        zs2 = int(torch.clamp(seed[1]*10, min=0, max=3))
        zs3 = int(torch.clamp(seed[2]*10, min=0, max=3))
        return zs1,zs2,zs3

    def update_noise(self,epsilon,pro_num):
        self.epsilon = epsilon
        self.pro_num = pro_num

def set_noise(epoch,adv_type):
    if 0<=epoch < 10:
        if adv_type=="fgsm":
            return 0.7,1
        else:
            return 0.5, 25
    elif 10<=epoch < 20:
        if adv_type=="fgsm":
            return 0.6,1
        else:
            return 0.4, 20
    elif 20<=epoch < 30:
        if adv_type=="fgsm":
            return 0.8,1
        else:
            return 0.6, 20
    elif 30<=epoch < 40:
        if adv_type=="fgsm":
            return 0.6,1
        else:
            return 0.3, 15
    
    elif 40<=epoch<= 50:
        if adv_type=="fgsm":
            return 0.5,1
        else:
            return 0.2, 10



if __name__=="__main__":
    data_file = os.path.join(args.data_path, args.data_set)
    HR10=[]
    NDCG10=[]
    advHR10=[]
    advNDCG10=[]
    train_data, test_data, user_num, item_num, train_mat = data_util.load_all(data_file)

    train_dataset = data_util.NCFData(train_data, item_num, train_mat, args.num_ng, True)
    test_dataset = data_util.NCFData(test_data, item_num, train_mat, 0, False)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_loader = data.DataLoader(test_dataset, batch_size=args.test_num_ng + 1, shuffle=False, num_workers=0, drop_last=True)


    model1 = MLP(user_num, item_num, args.embedding_dim, args.hidden_layer, args.dropout)
    model1.to(device)
    model2 = RAT_MLP(user_num, item_num, args.embedding_dim, args.hidden_layer, args.dropout,args.epsilon, args.pro_num,\
                args.enable_lat, args.layerlist, args.adv_type, args.adv_reg, args.norm,args.decay_factor, device)
    model2.to(device)
    loss_function = nn.BCEWithLogitsLoss()

    optimizer1 = optim.Adam(model1.parameters(), lr=args.lr)
    optimizer2 = optim.Adam(model2.parameters(), lr=args.lr)

    # writer = SummaryWriter() # for visualization

    ########################### TRAINING #####################################
    HR2, NDCG2 = evaluate2.metrics(model2, test_loader, args.top_k, device)
    print(f"Init model2: HR10={np.mean(HR2):.4f}, NDCG10={np.mean(NDCG2):.4f}")
    count1, best_hr1, count2, best_hr2 = 0, 0, 0, 0
    # model1.train()
    model2.train()
    if args.enable_lat:
        model2.grad_init()
    for epoch1 in range(args.epochs):
          # Enable dropout (if have).
        start_time = time.time()
        train_loader.dataset.ng_sample()

        for user, item, label in train_loader:
            user = user.to(device)
            item = item.to(device)
            label = label.float().to(device)

            optimizer1.zero_grad()
            prediction1 = model1(user, item)
            loss1 = loss_function(prediction1, label)
            loss1.backward()
            optimizer1.step()
            # writer.add_scalar('data/loss', loss.item(), count)
            count1 += 1
        HR1, NDCG1 = evaluate.metrics(model1, test_loader, args.top_k, device)
        HR10.append(np.mean(HR1))
        NDCG10.append(np.mean(NDCG1))
        elapsed_time = time.time() - start_time
        print("model1: The time elapse of epoch {:03d}".format(epoch1) + " is: " +
                time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR1), np.mean(NDCG1)))
        if HR1 > best_hr1:
            best_hr1, best_ndcg1, best_epoch1 = HR1, NDCG1, epoch1
        
    for epoch2 in range(args.epochs):   
        start_time = time.time()
        train_loader.dataset.ng_sample()
        if args.enable_lat:
            model2.update_seed() 

        if args.enable_lat:
            args.epsilon, args.pro_num = set_noise(epoch2,adv_type=args.adv_type)
            model2.update_noise(args.epsilon, args.pro_num)
        
        for user, item, label in train_loader:
            user = user.to(device)
            item = item.to(device)
            label = label.float().to(device)
            prediction2,loss2 = model2(user, item,label)

            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()
            if args.enable_lat:
                model2.save_grad()
            # writer.add_scalar('data/loss', loss.item(), count)
            count2 += 1
        HR2, NDCG2 = evaluate2.metrics(model2, test_loader, args.top_k, device) 
        advHR10.append(np.mean(HR2))
        advNDCG10.append(np.mean(NDCG2))
        elapsed_time = time.time() - start_time
        print("model2: The time elapse of epoch {:03d}".format(epoch2) + " is: " +
                time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR2), np.mean(NDCG2)))
        if HR2 > best_hr2:
            best_hr2, best_ndcg2, best_epoch2 = HR2, NDCG2, epoch2

    show_metric(args.epochs, HR10,NDCG10,advHR10,advNDCG10, 'HR@10', 'NDCG@10', args.adv_type)
    # show_curve(20, NDCG10, 'NDCG@10') 

    print("End model1. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(best_epoch1, best_hr1, best_ndcg1))
    print("End model2. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(best_epoch2, best_hr2, best_ndcg2))
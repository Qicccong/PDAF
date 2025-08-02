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
import matplotlib.pyplot as plt
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def parse_args():
    parser = argparse.ArgumentParser(description="Run ConvNCF.")
    parser.add_argument("--enable_lat", nargs="?", default=True)
    parser.add_argument("--epsilon", nargs="?", default=0.5)
    parser.add_argument("--alpha", nargs="?", default=1)
    parser.add_argument("--pro_num", nargs="?", default=25, choices=[1, 25], help="1 for fgsm and 10 for bim/pgd")
    parser.add_argument("--decay_factor", nargs="?", default=1.0)
    parser.add_argument("--layerlist", nargs="?", default="all")
    parser.add_argument("--adv", nargs="?", default=True)
    parser.add_argument("--adv_reg", nargs="?", default=1)
    parser.add_argument("--adv_type", nargs="?", default="mim", choices=['fgsm', 'bim', 'pgd','mim'])
    parser.add_argument("--norm", nargs="?", default="linf", choices=['linf', 'l2'])
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
    # plt.savefig("/data/chenhai-fwxz/DeepCF-PyTorch/result/%s_%s_%s.jpg" %(adv,title1, title2))



class ConvNCF(nn.Module):

    def __init__(self, user_count, item_count,device):
        super(ConvNCF, self).__init__()

        # some variables
        self.device = device
        self.user_count = user_count
        self.item_count = item_count

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


class RAT_ConvNCF(nn.Module):

    def __init__(self, user_count, item_count, enable_lat, layerlist, adv_type, adv_reg, norm,decay_factor,epsilon, pro_num,device):
        super(RAT_ConvNCF, self).__init__()
        self.epsilon = epsilon
        self.adv_type = adv_type
        self.norm = norm
        self.device = device
        self.adv_reg = adv_reg
        self.decay_factor = decay_factor
        self.pro_num = pro_num
        self.enable_lat = enable_lat
        self.seed1,_,_,_,_,_ = self.random()
        _,self.seed2,_,_,_,_ = self.random()
        _,_,self.seed3,_,_,_ = self.random()
        _,_,_,self.seed4,_,_ = self.random()
        _,_,_,_,self.seed5,_ = self.random()
        _,_,_,_,_,self.seed6 = self.random()
        self.maxseed = max(self.seed1, self.seed2,self.seed3,self.seed4,self.seed5,self.seed6)
        self.y_list = ['y0_reg','y1_reg','y2_reg','y3_reg','y4_reg','y5_reg','y6_reg']
        self.enable_list = [0 for i in range(7)]
        if enable_lat and layerlist != "all":
            self.layerlist = [int(x) for x in layerlist.split(',')]
            self.layerlist_digit = [int(x) for x in layerlist.split(',')]
        else:
            self.layerlist = "all"
            self.layerlist_digit = list(range(0,self.maxseed+1))
        # some variables
        self.user_count = user_count
        self.item_count = item_count

        # embedding setting
        self.embedding_size = 64

        self.P = nn.Embedding(self.user_count, self.embedding_size).to(self.device)
        self.Q = nn.Embedding(self.item_count, self.embedding_size).to(self.device)

        # cnn setting
        self.channel_size = 32
        self.kernel_size = 2
        self.strides = 2
        cnn_modules = []
        self.reg_size_list = list()
        y_index = 1
        self.register_buffer(self.y_list[0], torch.zeros([100, 64, 64]))
        for i in range(7):
            cnn_modules.append(nn.Conv2d(1, self.channel_size, self.kernel_size, stride=self.strides))
            cnn_modules.append(nn.ReLU())
            self.register_buffer(self.y_list[y_index], torch.zeros([1000, 32,32,32]))
            self.reg_size_list.append([1000, 32,32,32])
            cnn_modules.append(nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides))
            cnn_modules.append(nn.ReLU())
            self.register_buffer(self.y_list[2], torch.zeros([1000, 32,16,16]))
            self.reg_size_list.append([1000, 32,16,16])
            cnn_modules.append(nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides))
            cnn_modules.append(nn.ReLU())
            self.register_buffer(self.y_list[3], torch.zeros([1000, 32,8,8]))
            self.reg_size_list.append([1000, 32,8,8])
            cnn_modules.append(nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides))
            cnn_modules.append(nn.ReLU())
            self.register_buffer(self.y_list[4], torch.zeros([1000, 32,4,4]))
            self.reg_size_list.append([1000, 32,4,4])
            cnn_modules.append(nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides))
            cnn_modules.append(nn.ReLU())
            self.register_buffer(self.y_list[5], torch.zeros([1000, 32,2,2]))
            self.reg_size_list.append([1000, 32,2,2])
            cnn_modules.append(nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides))
            cnn_modules.append(nn.ReLU())
            self.register_buffer(self.y_list[6], torch.zeros([1000, 32,1,1]))
            self.reg_size_list.append([1000, 32,1,1])
            
        self.cnn = nn.Sequential(*cnn_modules)

        self.fc = nn.Linear(32, 1)
        self.choose_layer()
        

    def forward(self, user_ids, item_ids, is_pretrain):

        # convert float to int
        user_ids = list(map(int, user_ids))
        item_ids = list(map(int, item_ids))

        user_embeddings = self.P(torch.tensor(user_ids).to(self.device))
        item_embeddings = self.Q(torch.tensor(item_ids).to(self.device))

            
        if is_pretrain:
            # inner product
            prediction = torch.sum(torch.mul(user_embeddings, item_embeddings), dim=1).to(self.device)
        # else:
            
        #     interaction_map = torch.bmm(user_embeddings.unsqueeze(2), item_embeddings.unsqueeze(1)).to(self.device)
        #     self.y0 = interaction_map.view((-1, 1, self.embedding_size, self.embedding_size)).to(self.device)

            # cnn
            # self.y1 = self.cnn[0](self.y0).to(device)
            # self.z1 = self.cnn[1](self.y1).to(self.device)
            # self.y2 = self.cnn[2](self.z1).to(self.device) 
            # self.z2 = self.cnn[3](self.y2).to(self.device)
            # self.y3 = self.cnn[4](self.z2).to(self.device)
            # self.z3 = self.cnn[5](self.y3).to(self.device) 
            # self.y4 = self.cnn[6](self.z3).to(self.device)  
            # self.z4 = self.cnn[7](self.y4).to(self.device)
            # self.y5 = self.cnn[8](self.z4).to(self.device) 
            # self.z5 = self.cnn[9](self.y5).to(self.device)
            # self.y6 = self.cnn[10](self.z5).to(self.device)
            # self.z6 = self.cnn[11](self.y6).to(self.device)
            # feature_vec = self.z6.view((-1, 32)).to(self.device)

            # # fc
            # prediction = self.fc(feature_vec).to(self.device)

            # # fc
            # prediction = self.fc(feature_vec).to(self.device)
            # prediction1 = prediction.view((-1)).to(self.device)
        
       
        # if is_pretrain:
        #     # inner product
        #     prediction2 = torch.sum(torch.mul(user_embeddings, item_embeddings), dim=1).to(self.device)
        else:
            interaction_map = torch.bmm(user_embeddings.unsqueeze(2), item_embeddings.unsqueeze(1)).to(self.device)
            self.y0 = interaction_map.view((-1, 1, self.embedding_size, self.embedding_size)).to(self.device)
            self.y1 = self.cnn[0](self.y0).to(device)
            if self.enable_lat and 1 in self.enable_list1:
                self.y1.retain_grad()
            self.y1_add = self.y1.add(self.y1_reg.data).to(device)
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
        self.z1 = self.cnn[1](self.y1_add).to(self.device)
        
        self.y2 = self.cnn[2](self.z1).to(self.device) 
        if self.enable_lat and 2 in self.enable_list2:
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
        self.z2 = self.cnn[3](self.y2_add).to(self.device)  #torch.Size([100, 128])

        self.y3 = self.cnn[4](self.z2).to(self.device)   
        if self.enable_lat and 3 in self.enable_list3:
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
        self.z3 = self.cnn[5](self.y3_add).to(self.device) 

        self.y4 = self.cnn[6](self.z3).to(self.device)  
        if self.enable_lat and 4 in self.enable_list4:
            self.y4.retain_grad()
            self.y4_add = self.y4.add(self.y4_reg.data)
            if self.norm =="linf":
                    delta = torch.clamp(self.y4_add-self.y4, -self.epsilon, self.epsilon).to(self.device) 
                    self.y4_add = self.y4.add(delta).to(self.device)
            if self.norm =="l2":
                delta = self.y4_add-self.y4
                normVal = torch.norm(delta.view(100, -1), 2, 1)
                mask = normVal<=self.epsilon
                scaling = self.epsilon/normVal
                scaling[mask] = 1
                delta = delta*scaling.view(100, 1)
                self.y4_add = self.y4.add(delta).to(self.device)
        else:
            self.y4_add = self.y4
            self.y4_add = self.y4_add.to(self.device)
        self.z4 = self.cnn[7](self.y4_add)

        self.y5 = self.cnn[8](self.z4).to(self.device)  
        if self.enable_lat and 5 in self.enable_list5:
            self.y5.retain_grad()
            self.y5_add = self.y5.add(self.y5_reg.data)
            if self.norm =="linf":
                    delta = torch.clamp(self.y5_add-self.y5, -self.epsilon, self.epsilon).to(self.device) 
                    self.y5_add = self.y5.add(delta).to(self.device)
            if self.norm =="l2":
                delta = self.y5_add-self.y5
                normVal = torch.norm(delta.view(100, -1), 2, 1)
                mask = normVal<=self.epsilon
                scaling = self.epsilon/normVal
                scaling[mask] = 1
                delta = delta*scaling.view(100, 1)
                self.y5_add = self.y5.add(delta).to(self.device)
        else:
            self.y5_add = self.y5
            self.y5_add = self.y5_add.to(self.device)
        self.z5 = self.cnn[9](self.y5_add)

        self.y6 = self.cnn[10](self.z5).to(self.device)  
        if self.enable_lat and 6 in self.enable_list6:
            self.y6.retain_grad()
            self.y6_add = self.y6.add(self.y6_reg.data)
            if self.norm =="linf":
                    delta = torch.clamp(self.y6_add-self.y6, -self.epsilon, self.epsilon).to(self.device) 
                    self.y6_add = self.y6.add(delta).to(self.device)
            if self.norm =="l2":
                delta = self.y6_add-self.y6
                normVal = torch.norm(delta.view(100, -1), 2, 1)
                mask = normVal<=self.epsilon
                scaling = self.epsilon/normVal
                scaling[mask] = 1
                delta = delta*scaling.view(100, 1)
                self.y6_add = self.y6.add(delta).to(self.device)
        else:
            self.y6_add = self.y6
            self.y6_add = self.y6_add.to(self.device)
        self.z6 = self.cnn[11](self.y6_add).to(self.device)
        feature_vec = self.z6.view((-1, 32)).to(self.device)

        # fc
        prediction = self.fc(feature_vec).to(self.device)
        prediction = prediction.view((-1)).to(self.device)


        return prediction

    def grad_init(self):
        if self.adv_type=="naive":
            for i in range(1,self.maxseed+1): 
                exec('self.y{}_reg.data = torch.randn(self.y{})'.format(i,i))
        elif self.adv_type=="fgsm" or self.adv_type=="bim" or self.adv_type=="mim":
            for i in range(1,self.maxseed+1): 
                exec('self.y{}_reg.data = torch.zeros_like(self.y{}).detach()'.format(i,i))
        elif self.adv_type=="pgd":
            for i in range(1,self.maxseed+1):
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
            self.enable_list4 = list(range(0, self.seed4+1))
            self.enable_list5 = list(range(0, self.seed5+1))
            self.enable_list6 = list(range(0, self.seed6+1))   # all True
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

            if 4 in self.enable_list4:
                if self.norm=="linf" and self.adv_type!="mim":
                    self.y4_reg.data = (self.epsilon / self.pro_num) * (self.y4.grad.sign())
                elif self.norm=="l2" and self.adv_type!="mim":
                    self.y4_reg.data = (self.epsilon / self.pro_num) * (self.y4.grad / cal_lp_norm(self.y4.grad,p=2,dim_count=len(self.y4_add.grad.size())))
                elif self.adv_type=="mim":
                    momentum4 = torch.zeros_like(self.y4).detach()
                    grad4 = self.y4.grad
                    grad_norm4 = torch.norm(nn.Flatten()(grad4), p=1, dim=1)
                    grad4 = grad4 / grad_norm4.view([-1]+[1]*(len(grad4.shape)-1))
                    grad4 = grad4 + self.decay_factor * momentum4
                    momentum4 = grad4
                    if self.norm=="linf":
                        self.y4_reg.data = (self.epsilon / self.pro_num) * (grad4.sign())

                    elif self.norm=="l2":
                        self.y4_reg.data = (self.epsilon / self.pro_num) * (grad4 / cal_lp_norm(grad4,p=2,dim_count=len(self.y4.grad.size())))

            if 5 in self.enable_list5:
                if self.norm=="linf" and self.adv_type!="mim":
                    self.y5_reg.data = (self.epsilon / self.pro_num) * (self.y5.grad.sign())
                elif self.norm=="l2" and self.adv_type!="mim":
                    self.y5_reg.data = (self.epsilon / self.pro_num) * (self.y5.grad / cal_lp_norm(self.y5.grad,p=2,dim_count=len(self.y5_add.grad.size())))
                elif self.adv_type=="mim":
                    momentum5 = torch.zeros_like(self.y5).detach()
                    grad5 = self.y5.grad
                    grad_norm5 = torch.norm(nn.Flatten()(grad5), p=1, dim=1)
                    grad5 = grad5 / grad_norm5.view([-1]+[1]*(len(grad5.shape)-1))
                    grad5 = grad5 + self.decay_factor * momentum5
                    momentum5 = grad5
                    if self.norm=="linf":
                        self.y5_reg.data = (self.epsilon / self.pro_num) * (grad5.sign())

                    elif self.norm=="l2":
                        self.y5_reg.data = (self.epsilon / self.pro_num) * (grad5 / cal_lp_norm(grad5,p=2,dim_count=len(self.y5.grad.size())))

            if 6 in self.enable_list6:
                if self.norm=="linf" and self.adv_type!="mim":
                    self.y6_reg.data = (self.epsilon / self.pro_num) * (self.y6.grad.sign())
                elif self.norm=="l2" and self.adv_type!="mim":
                    self.y6_reg.data = (self.epsilon / self.pro_num) * (self.y6.grad / cal_lp_norm(self.y6.grad,p=2,dim_count=len(self.y6_add.grad.size())))
                elif self.adv_type=="mim":
                    momentum6 = torch.zeros_like(self.y6).detach()
                    grad6 = self.y6.grad
                    grad_norm6 = torch.norm(nn.Flatten()(grad6), p=1, dim=1)
                    grad6 = grad6 / grad_norm6.view([-1]+[1]*(len(grad6.shape)-1))
                    grad6 = grad6 + self.decay_factor * momentum6
                    momentum6 = grad6
                    if self.norm=="linf":
                        self.y6_reg.data = (self.epsilon / self.pro_num) * (grad6.sign())

                    elif self.norm=="l2":
                        self.y6_reg.data = (self.epsilon / self.pro_num) * (grad6 / cal_lp_norm(grad6,p=2,dim_count=len(self.y6.grad.size())))
    
    def update_seed(self):
        self.seed1,self.seed2,self.seed3,self.seed4,self.seed5,self.seed6 = self.random()

    def random(self):
        seed = torch.rand(6)*0.7
        zs1= int(torch.clamp(seed[0]*10, min=0, max=6))
        zs2 = int(torch.clamp(seed[1]*10, min=0, max=6))
        zs3 = int(torch.clamp(seed[2]*10, min=0, max=6))
        zs4 = int(torch.clamp(seed[3]*10, min=0, max=6))
        zs5 = int(torch.clamp(seed[4]*10, min=0, max=6))
        zs6 = int(torch.clamp(seed[5]*10, min=0, max=6))
        return zs1,zs2,zs3,zs4,zs5,zs6

    def update_noise(self,epsilon,pro_num):
        self.epsilon = epsilon
        self.pro_num = pro_num


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


def train(model, adversarial):
    lr = 0.05
    epoches = 300
    batch_size = 100
    losses = []
    accuracies = []
    H10 = []
    NDCG10 = []
    bestHr10, bestNdcg10, bestepoch= 0,0, -1
    model.train()
    
    optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1, last_epoch=-1)

    train_loader = Data.DataLoader(yelp.train_group, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
    
    bpr_loss = BPRLoss().to(device)

    for epoch in range(epoches):    

        
        total_loss = 0
        total_acc = 0
        try:
            with tqdm(train_loader) as t:
                for batch_idx, train_data in enumerate(t):                    
        #             train_data = Variable(train_data.to(self.device))
                    user_ids = Variable(train_data[:, 0].to(device))
                    pos_item_ids = Variable(train_data[:, 1].to(device))
                    neg_item_ids = Variable(train_data[:, 2].to(device))
                    
                    optimizer.zero_grad()
                    


                    if epoch < 100:    
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
        print('Epoch:', epoch, 'Train loss:', losses[-1], 'Train acc:', accuracies[-1])  
        # if epoch % 50 == 0 and epoch != 0:      
        #     show_curve(losses, 'train loss')
        #     show_curve(accuracies, 'train acc')
        #     evaluate()
        # if epoch == 3 or epoch == 20:
        hr10, ndcg10 = evaluate(epoch,model,adversarial)
        H10.append(hr10)
        NDCG10.append(ndcg10)
        scheduler.step()
        if hr10 > bestHr10:
                bestHr10, bestNdcg10, bestepoch= hr10, ndcg10, epoch             
    print(f"Best epoch {bestepoch+1}: HR10={bestHr10:.4f}, NDCG10={bestNdcg10:.4f}")
    return H10, NDCG10
    
def train2(model):
    args = parse_args()
    lr = 0.05
    epoches = 1500
    batch_size = 100
    losses = []
    accuracies = []
    H10 = []
    NDCG10 = []
    bestHr10, bestNdcg10, bestepoch= 0,0, -1
    hr10_init, ndcg10_init = evaluate(1, model2)
    print(f"Init model2: HR10={hr10_init:.4f}, NDCG10={ndcg10_init:.4f}")
    model.train()
    if args.enable_lat:
        model.grad_init()
    optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1, last_epoch=-1)
    train_loader = Data.DataLoader(yelp.train_group, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
    
    bpr_loss = BPRLoss().to(device)

    for epoch in range(epoches):    

        
        total_loss = 0
        total_acc = 0
        try:
            with tqdm(train_loader) as t:
                if args.enable_lat:
                    model.update_seed() 

                if args.enable_lat:
                    args.epsilon, args.pro_num = set_noise(epoch,adv_type=args.adv_type)
                    model.update_noise(args.epsilon, args.pro_num)
                
                for batch_idx, train_data in enumerate(t):                    
        #             train_data = Variable(train_data.to(self.device))
                    user_ids = Variable(train_data[:, 0].to(device))
                    pos_item_ids = Variable(train_data[:, 1].to(device))
                    neg_item_ids = Variable(train_data[:, 2].to(device))
                    
                    optimizer.zero_grad()
                    


                    if epoch < 100:    
                        # pretrain bpr
                        pos_preds = model(user_ids, pos_item_ids, True)
                        neg_preds = model(user_ids, neg_item_ids, True)                        
                    else:
                        # train convncf                        
                        pos_preds = model(user_ids, pos_item_ids, False)
                        neg_preds = model(user_ids, neg_item_ids, False)
                        
                    loss = bpr_loss(pos_preds, neg_preds)                  
                    total_loss += loss.item()                    
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    if args.enable_lat:
                        model.save_grad() 
                    
                    accuracy = float(((pos_preds - neg_preds) > 0).sum()) / float(len(pos_preds))
                    total_acc += accuracy
        #             if batch_idx % 20 == 0:
        #                 print('Epoch:', epo.ch, 'Batch:', batch_idx, 'Train loss:', losses[-1], 'Train acc:', accuracies[-1])                                
        except KeyboardInterrupt:
            t.close()
            raise
        
        losses.append(total_loss / (batch_idx + 1))
        accuracies.append(total_acc / (batch_idx + 1))
        print('Epoch:', epoch, 'Train loss:', losses[-1], 'Train acc:', accuracies[-1])  
        # if epoch % 50 == 0 and epoch != 0:      
        #     show_curve(losses, 'train loss')
        #     show_curve(accuracies, 'train acc')
        #     evaluate()
        # if epoch == 3 or epoch == 20:
        hr10, ndcg10 = evaluate(epoch,model)
        H10.append(hr10)
        NDCG10.append(ndcg10)
        scheduler.step()
        if hr10 > bestHr10:
                bestHr10, bestNdcg10, bestepoch= hr10, ndcg10, epoch             
    print(f"Best epoch {bestepoch+1}: HR10={bestHr10:.4f}, NDCG10={bestNdcg10:.4f}")

    return H10, NDCG10            

def evaluate(epoch, model):
    model.eval()
    hr5 = []
    hr10 = []
    hr20 = []
    ndcg5 = []
    ndcg10 = []
    ndcg20 = []
    user_count = len(yelp.test_negative)
    try:
        with tqdm(range(user_count)) as t:
            for u in t:
                item_ids = torch.tensor(yelp.test_negative[u]).to(device)
                user_ids = torch.tensor([u] * len(item_ids)).to(device)
                # if adversarial:
                #     predictions = model(user_ids, item_ids, False)
                # else:
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
            # show_metric(epoch, hr10, 'HR@10')
            # show_metric(epoch, ndcg10, 'NDCG@10')
            # show_metric(hr20, 'HR@20')
            # show_metric(ndcg20, 'NDCG@20')            
            # print('HR@5:', sum(hr5) / len(hr5))
            # print('NDCG@5:', sum(ndcg5) / len(ndcg5))
            hr_10 = sum(hr10) / len(hr10)
            
            ndcg_10 = sum(ndcg10) / len(ndcg10)
            
            print('HR@10:', hr_10)
            print('NDCG@10:', ndcg_10)
        
            # print('HR@20:', sum(hr20) / len(hr20))
            # print('NDCG@20:', sum(ndcg20) / len(ndcg20))
    except KeyboardInterrupt:
        t.close()
        raise
    return hr_10, ndcg_10


def scoreK(topi, k):
    hr = 1.0 if 999 in topi[0:k] else 0.0
    if hr:
        ndcg = math.log(2) / math.log(topi.tolist().index(999) + 2)
    else:
        ndcg = 0
    # auc = 1 - (position * 1. / negs)
    return hr, ndcg#, auc


if __name__ == '__main__':
    # torch.set_num_threads(12)
    args = parse_args()
    print('Data loading...')
    yelp = load_dataset.Load_Yelp('./Data/lastfm.train.rating', './Data/lastfm.test.rating', './Data/lastfm.test.negative')
    print('Data loaded')

    print('=' * 50)

    print('Model initializing...')
    # model1 = ConvNCF(int(max(yelp.train_group[:, 0])) + 1, int(max(yelp.train_group[:, 1])) + 1, device).to(device)
    model2 = RAT_ConvNCF(int(max(yelp.train_group[:, 0])) + 1, int(max(yelp.train_group[:, 1])) + 1, args.enable_lat, args.layerlist, args.adv_type, args.adv_reg, args.norm,args.decay_factor,args.epsilon, args.pro_num, device).to(device)
    print('Model initialized')

    print('=' * 50)

    print('Model training...')
    # H10,ndcg10=train(model1, False)
    HR10, NDCG10=train2(model2)
    print('Model trained')
    # show_metric(1500, HR10,ndcg10,HR10,NDCG10, 'HR@10', 'NDCG@10', args.adv_type)
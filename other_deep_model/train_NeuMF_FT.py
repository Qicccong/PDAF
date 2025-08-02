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
from utils_pdaf import *
# from utils_awp import AdvWeightPerturb1
import data_util
import evaluate
import evaluate1
import evaluate2
from collections import OrderedDict
# import GMF
# import MLP
# from train_NeuMF import NeuMF

EPS = 1E-20
random.seed(123)
def update_seed():
    T = random.uniform(0, 1)
    if T>0.5:
        return [1, 1, 1, 1, 1, 1, 1, 1, 1]
    else:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0] 

def diff_in_weights(model, proxy,layers):
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
        if len(old_w.size()) <= 1:
            continue
        if 'weight' in old_k:

            # if  old_k == layers:
            if  layers in old_k:
                diff_w = new_w - old_w
                diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
    return diff_dict


def add_into_weights(model, diff,coeff=1.0,ratio=0.5): # 使用pgd更新模型参数
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            random_float = random.uniform(0, 1)
            if name in names_in_diff and random_float < ratio:
                param.add_(coeff * diff[name])

class AdvWeightPerturb1(object):
    def __init__(self, model, proxy, proxy_optim, gamma):
        super(AdvWeightPerturb1, self).__init__()
        self.model = model
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.gamma = gamma

    def calc_awp(self, inputs_adv_u, inputs_adv_i,inputs_adv_lab,layers):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()
        criterion = torch.nn.BCELoss()
        # loss = - F.cross_entropy(self.proxy(inputs_adv_u, inputs_adv_i), inputs_adv_lab)
        # output,_ = self.proxy(inputs_adv_u, inputs_adv_i)
        *_,output = self.proxy(inputs_adv_u, inputs_adv_i,inputs_adv_lab)
        yc = output.squeeze()
        loss = -criterion(yc, inputs_adv_lab.float())
    
        # loss = - criterion(output, inputs_adv_lab)
        self.proxy_optim.zero_grad()
        loss.backward()
        self.proxy_optim.step()

        # the adversary weight perturb
        diff = diff_in_weights(self.model, self.proxy,layers)
        return diff

    def perturb(self, diff):
        add_into_weights(self.model, diff, coeff=1.0 * self.gamma)

    def restore(self, diff):
        add_into_weights(self.model, diff, coeff=-1.0 * self.gamma)


parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate")
parser.add_argument("--batch_size", type=int, default=100, help="batch size for training")
parser.add_argument("--epochs", type=int, default=100, help="training epoches")
parser.add_argument("--top_k", type=int, default=10, help="compute metrics@top_k")
parser.add_argument("--embedding_dim_GMF", type=int, default=32, help="dimension of embedding in GMF submodel")
parser.add_argument("--embedding_dim_MLP", type=int, default=32, help="dimension of embedding in MLP submodel")
parser.add_argument("--hidden_layer_MLP", type=list, default=[64,32,16,8], help="hidden layers in MLP")
parser.add_argument("--use_pretrained", action="store_true",default=False, help="use pretrained model to initialize weights")
parser.add_argument("--num_ng", type=int, default=4, help="sample negative items for training")
parser.add_argument("--test_num_ng", type=int, default=99, help="sample part of negative items for testing")
parser.add_argument("--data_set", type=str, default="lastfm", help="data set. 'ml-1m' or 'pinterest-20'")
parser.add_argument('--awp-gamma', default=0.005, type=float)   # 0.001,0.005
parser.add_argument("--data_path", type=str, default="Data/")
parser.add_argument("--model_path", type=str, default="/results/model")
parser.add_argument("--out", default=False, help="save model or not")
# parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
parser.add_argument("--enable_lat", nargs="?", default=True)
# parser.add_argument("--epsilon", nargs="?", default=0.5)
parser.add_argument("--alpha", nargs="?", default=0.008/10)
parser.add_argument("--num_steps", nargs="?", default=10)  #25
# parser.add_argument("--pro_num", nargs="?", default=1, choices=[1, 25], help="1 for fgsm and 10 for bim/pgd")
parser.add_argument("--decay_factor", nargs="?", default=1.0)
parser.add_argument("--layerlist", nargs="?", default="all")
parser.add_argument("--adv", nargs="?", default=True)
parser.add_argument("--adv_reg", nargs="?", default=1)
parser.add_argument("--reg", nargs="?", default=1e-3)
parser.add_argument("--adv_type", nargs="?", default="fgsm", choices=['fgsm', 'bim', 'pgd','mim'])
parser.add_argument('--device', default="cuda", type=str, help='device')
parser.add_argument('--epsilon', default=0.008, type=float)
parser.add_argument('--attack', default="fgsm")
# parser.add_argument("--norm", nargs="?", default="linf", choices=['linf', 'l2'])
args = parser.parse_args()

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
            # loss,_,_,_,_,_,_,_= model(ui, ii,lbl)
            loss,*_= model(ui, ii,lbl)
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
                    # loss1,_,_,_,_,_,_,outputs= cloned_model(ui, ii,lbl)
                    loss1,*_,outputs= cloned_model(ui, ii,lbl)
                    outputs = outputs.squeeze()
                    loss = - criterion(outputs, lbl.float())
                    loss.backward()
                    optimizer.step()
                    cal_num = cal_num +1
                    # if  cal_num >= num_tcal:
                    #         break
                cloned_model.eval()
                sd = cloned_model.state_dict()

                # 使用维度无关的计算方法
                for layer_name in sd.keys():
                    if sd[layer_name] is not init_param:  # 跳过非目标层的参数
                        continue
                        
                    # 获取当前参数值
                    param_value = sd[layer_name]
                    
                    # 计算差值
                    diff = param_value - init_param
                    
                    # 获取参数形状
                    param_shape = param_value.shape
                    
                    # 计算范数（使用原始形状）
                    normVal1 = torch.linalg.norm(diff)
                    normVal2 = torch.linalg.norm(init_param)
                    
                    # 计算缩放因子
                    scaling_factor = normVal2 / normVal1 * epsilon
                    
                    # 避免无穷大
                    if torch.isinf(scaling_factor):
                        scaling_factor = torch.tensor(0.0)
                    
                    # 应用缩放因子
                    diff = diff * scaling_factor
                    
                    # 更新参数
                    sd[layer_name] = init_param + diff

                # diff = sd[layer_name] - init_param
                # times = torch.linalg.norm(diff)/torch.linalg.norm(init_param)


                # size_para = init_param.size(0)
                
                # normVal1 = torch.norm(diff.view(size_para, -1), 2, 1)
                # normVal2 = torch.norm(init_param.view(size_para, -1), 2, 1)
                # scaling = normVal2/normVal1 * epsilon   # 算出一次噪声需要调整的倍数
                # scaling[scaling == float('inf')] = 0
                # diff = diff*scaling.view(size_para, 1)

                # sd[layer_name] = deepcopy(init_param + diff)


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
                        # loss1,_,_,_,_,_,_,outputs= cloned_model(ui, ii,lbl)
                        loss1,*_,outputs= cloned_model(ui, ii,lbl)
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


def train_model(args, model, dataloader, optimizer, criterion):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    start_time = time.time()
    losses = AverageMeter("Loss")

    for ui, ii, lbl in dataloader:
        ui, ii, lbl = ui.to(device), ii.to(device), lbl.to(device)  
        lbl = lbl.float()  # 确保标签是浮点数

        awp = args.awp_adversary.calc_awp(inputs_adv_u=ui, inputs_adv_i=ii,inputs_adv_lab=lbl,layers = args.layer)
        args.awp_adversary.perturb(awp)

        loss,*_= model(ui, ii,lbl)

        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        args.awp_adversary.restore(awp)
        
    return train_loss / len(dataloader)

class NeuMF(nn.Module):
    def __init__(self, user_num, item_num, embedding_dim_GMF,embedding_dim_MLP, hidden_layer_MLP,
                 dropout,device, GMF_model=None,MLP_model=None):
        super(NeuMF, self).__init__()
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
        self.device = device
        self.dropout = dropout
        self.GMF_model = GMF_model
        self.MLP_model = MLP_model
        # self.criterion = torch.nn.BCELoss()  # 添加损失函数
        self.criterion = torch.nn.BCEWithLogitsLoss()
        
        self.embed_user_GMF = nn.Embedding(user_num, embedding_dim_GMF).to(self.device)
        self.embed_item_GMF = nn.Embedding(item_num, embedding_dim_GMF).to(self.device)
        self.embed_user_MLP = nn.Embedding(user_num, embedding_dim_MLP).to(self.device)
        self.embed_item_MLP = nn.Embedding(item_num, embedding_dim_MLP).to(self.device)

        self.MLP_layers = nn.ModuleList()
        self.num_layers = len(hidden_layer_MLP)
        self.relu_outputs = []  # 存储ReLU层的输出

        for i in range(self.num_layers):
            # Dropout 层
            dropout_layer = nn.Dropout(p=self.dropout)
            self.MLP_layers.append(dropout_layer)
            # Linear 层
            if i == 0:
                linear_layer = nn.Linear(embedding_dim_MLP*2, hidden_layer_MLP[0])
            else:
                linear_layer = nn.Linear(hidden_layer_MLP[i-1], hidden_layer_MLP[i])
            self.MLP_layers.append(linear_layer)
            # ReLU 层
            relu_layer = nn.ReLU()
            self.MLP_layers.append(relu_layer)

        # 预测层
        self.predict_layer = nn.Linear(hidden_layer_MLP[-1] + embedding_dim_GMF, 1)

        if not args.use_pretrained:
            nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
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
        else:
            # embedding layers
            self.embed_user_GMF.weight.data.copy_(self.GMF_model.embed_user.weight)
            self.embed_item_GMF.weight.data.copy_(self.GMF_model.embed_item.weight)
            self.embed_user_MLP.weight.data.copy_(self.MLP_model.embed_user.weight)
            self.embed_item_MLP.weight.data.copy_(self.MLP_model.embed_item.weight)

            # mlp layers
            for (m1, m2) in zip(self.MLP_layers, self.MLP_model.MLP_layers):
                if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                    m1.weight.data.copy_(m2.weight)
                    m1.bias.data.copy_(m2.bias)

            # predict layers
            predict_weight =  self.MLP_model.predict_layer.weight
            precit_bias = self.MLP_model.predict_layer.bias

            self.predict_layer.weight.data.copy_(0.5 * predict_weight)
            self.predict_layer.bias.data.copy_(0.5 * precit_bias)

    def forward(self, user, item, label):
        # GMF计算
        embed_user_GMF = self.embed_user_GMF(user).to(self.device)
        embed_item_GMF = self.embed_item_GMF(item).to(self.device)
        output_GMF = embed_user_GMF * embed_item_GMF

        embed_user_MLP = self.embed_user_MLP(user).to(self.device)
        embed_item_MLP = self.embed_item_MLP(item).to(self.device)
        interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1).to(self.device)
        # output_MLP = self.MLP_layers(interaction).to(self.device)
        # concat = torch.cat((output_GMF, output_MLP), -1).to(self.device)

        # prediction = self.predict_layer(concat)
        # return prediction.view(-1)
        
        # 逐层计算并保存中间输出
        x = interaction
        mlp_outputs = []  # 存储MLP中间层输出
        # 遍历所有层（Dropout->Linear->ReLU 为一组）
        for i in range(0, len(self.MLP_layers), 3):
            # Dropout
            x = self.MLP_layers[i](x)
            # Linear
            x = self.MLP_layers[i+1](x)
            # ReLU (保存此层输出)
            x = self.MLP_layers[i+2](x)
            x.requires_grad_(True)
            x.retain_grad()
            mlp_outputs.append(x)
        
        output_MLP = x

        # 合并路径
        concat = torch.cat((output_GMF, output_MLP), -1).to(self.device)

        # 全连接层预测
        prediction = torch.sigmoid(self.predict_layer(concat).view(-1))
        prediction = torch.clamp(prediction, 1e-8, 1 - 1e-8)

        logits = self.predict_layer(concat).view(-1)

        if isinstance(label, bool):
            return torch.sigmoid(logits)


        label = label.to(logits.device)
        loss = self.criterion(logits, label.float())
        prediction = torch.sigmoid(logits)

        # 预测
        # prediction = torch.sigmoid(self.predict_layer(concat)).view(-1)
        # 计算损失
        # loss = self.criterion(prediction, label.float())
        
        # 返回损失和中间层输出
        return loss, *mlp_outputs, prediction

if __name__=="__main__":
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("格式化时间:", formatted_time)
    print(f"mine_lr={args.lr}, dataset:{args.data_set}, gamma:{args.awp_gamma}")
    data_file = os.path.join(args.data_path, args.data_set)
    HR10=[]
    NDCG10=[]
    advHR10=[]
    advNDCG10=[]
    save_dir = f'pretrain/NeuMF_FT/{args.data_set}_{args.lr}_{args.awp_gamma}/'
    for path in [save_dir]:
        if not os.path.isdir(path):
            os.makedirs(path)
    logger = create_logger(save_dir+'output.log')
    logger.info(args)
    args.logger = logger

    train_data, test_data, user_num, item_num, train_mat = data_util.load_all(data_file)
    dataset = Dataset(args.data_path + args.data_set)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    
    # 计算数据集总元素数
    total_possible_interactions = user_num * item_num
    num_interactions = train.nnz  # 非零元素数量（正确的交互计数）
    sparsity = 1 - (num_interactions / total_possible_interactions)
    # 根据稀疏度动态调整负采样数量
    nNeg = int(5 + 15 * sparsity)  # 稀疏度越高，负样本越多
    # args.nNeg = nNeg
    args.num_ng = 4

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

    # train_dataset = data_util.NCFData(train_data, item_num, train_mat, args.num_ng, True)
    # test_dataset = data_util.NCFData(test_data, item_num, train_mat, 0, False)

    userInput, itemInput, labels = get_train_instances(train, args.num_ng)
    dst = BatchDataset(userInput, itemInput, labels)
    train_loader = data.DataLoader(dst, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_loader = data.DataLoader(dst, batch_size=args.test_num_ng + 1, shuffle=False, num_workers=0, drop_last=True)
    # for batch in train_loader:
    #     ui, ii, lbl = batch
    #     lbl = lbl.float()  # 统一转换

    GMF_model = None
    MLP_model = None
    model1 = NeuMF(user_num, item_num, 32, 32, [64,32,16,8], 0.0, device, GMF_model,MLP_model)
    proxy_adv = NeuMF(user_num, item_num, 32, 32, [64,32,16,8], 0.0, device, GMF_model,MLP_model)
    model1.to(device)
    proxy_adv.to(device)
    loss_function = nn.BCEWithLogitsLoss()
    # create model
    logger.info('==> Building model...')

    # 加载预训练的参数 RAWP
    model_path = args.model_path
    original_state_dict = torch.load(model_path, weights_only=False)
    new_state_dict111 = {}
    for key, value in original_state_dict.items():
        if 'reg' not in key:
            new_state_dict111[key] = value
    model1.load_state_dict(new_state_dict111) # 
    del new_state_dict111
    del original_state_dict
    torch.cuda.empty_cache()  # 释放显存

    model1 = model1.to(device)
    init_sd = deepcopy(model1.state_dict())  # 保存模型的初始状态：
    
    # init_sd = init_sd.to(device)

    if args.use_pretrained:
        optimizer1 = optim.SGD(model1.parameters(), lr=args.lr)
        proxy_opt = torch.optim.Adam(proxy_adv.parameters(), lr=0)
    else:
        optimizer1 = optim.Adam(model1.parameters(), lr=args.lr)
        proxy_opt = torch.optim.Adam(proxy_adv.parameters(), lr=0)

    lr_decays = [int(args.epochs * 0.5), int(args.epochs * 0.75)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer1, lr_decays, gamma=0.5, last_epoch=-1)

    awp_adversary = AdvWeightPerturb1(model=model1, proxy=proxy_adv, proxy_optim=proxy_opt, gamma=args.awp_gamma)
    criterion = torch.nn.BCELoss()
    best_hr1, best_ndcg1, best_epoch, best_performence = 0, 0, 0, 0
    
    model1.train()
    args.trainloader = train_loader
    args.optimizer = optimizer1
    # args.epsilon = epsilon
    args.testRatings = testRatings
    args.testNegatives = testNegatives
    args.topK = 10
    args.topK1 = 20
    args.topK2 = 50
    args.topK3 = 100

    sorted_layer_sharpness = layer_sharpness(args, model1, train_loader, 0.008)
    # sorted_items = sorted(sorted_layer_sharpness.items(), key=lambda x: x[1].item())
    sorted_items = sorted(sorted_layer_sharpness.items(), key=lambda x: x[1])
    min_layer_name = sorted_items[-1][0]
    args.layer = min_layer_name

    logger.info(args.layer)
    assert args.layer is not None
    for name, param in model1.named_parameters():  
        param.requires_grad = False
        if args.layer in name:
            param.requires_grad = True  

    for rounds_1 in range(20):
        
        hits10, ndcgs10, maps10, mrrs10 = evaluate1.evaluate_model(model1, testRatings, testNegatives, K=10, K1=20, K2=50,K3=100, num_thread=1, device=device)
        HR1, NDCG1, map10,mrr10 = np.array(hits10).mean(), np.array(ndcgs10).mean(),np.array(maps10).mean(), np.array(mrrs10).mean()
        robust_HR10 = evaluate_adv.evalulate_robustness(args, model1, train_loader, optimizer1, args.epsilon, testRatings, testNegatives, 10, 20, 50, 100)
        logger.info(f"epoch: {rounds_1} init: HR10= {HR1:.4f}, NDCG10= {NDCG1:.4f}, robust HR10= {robust_HR10:.4f}")
        label_train_true = 0
        
        temp_awp_gamma = args.awp_gamma
        args.awp_adversary = AdvWeightPerturb1(model=model1, proxy=proxy_adv, proxy_optim=proxy_opt, gamma=temp_awp_gamma)
        train_loss = train_model(args, model1, train_loader, optimizer1, loss_function)   # criterion
        logger.info("==> Train loss: {:.4f}".format(train_loss))

        hits10, ndcgs10, maps10, mrrs10 = evaluate1.evaluate_model(model1, testRatings, testNegatives, K=10, K1=20, K2=50,K3=100, num_thread=1, device=device)
        HR1, NDCG1, map10,mrr10 = np.array(hits10).mean(), np.array(ndcgs10).mean(),np.array(maps10).mean(), np.array(mrrs10).mean()
        robust_HR10 = evaluate_adv.evalulate_robustness(args, model1, train_loader, optimizer1, args.epsilon, testRatings, testNegatives, 10, 20, 50, 100)
        HR10.append(np.mean(HR1))
        NDCG10.append(np.mean(NDCG1))

        if rounds_1 == 0:
            args.init_HR10 = robust_HR10

        modelPath = save_dir + f"{args.data_set}_lr-{args.lr}_g-{args.awp_gamma}_NeuMF_FT.pth"
        if robust_HR10 > best_hr1: 
            logger.info(f"find better: epoch: {rounds_1} init: HR10= {HR1:.4f}, NDCG10= {NDCG1:.4f}, robust HR10= {robust_HR10:.4f}")
            best_hr1 = robust_HR10 
            best_ndcg1 = NDCG1
            best_epoch = rounds_1
            logger.info('==> Saving best params...')
            best_model_path = modelPath
            if os.path.exists(best_model_path):
                os.remove(best_model_path)
            state = {
                'model': model1.state_dict(),
                'acc': best_hr1,
                'epoch': rounds_1,
            }
            torch.save(state, best_model_path)
            label_train_true = 1
        scheduler.step()

    if label_train_true > 0:
        checkpoint = torch.load(best_model_path, weights_only=False)
        model1.load_state_dict(checkpoint["model"])
        del checkpoint
    torch.cuda.empty_cache()  # 释放显存
    model_copy = deepcopy(model1.state_dict())
    for key, value in model_copy.items():
        if hasattr(value, 'to'):
            model_copy[key] = value
        else:
            model_copy[key] = value
    torch.cuda.empty_cache()  # 释放显存

    hits10, ndcgs10, maps10, mrrs10 = evaluate1.evaluate_model(model1, testRatings, testNegatives, K=10, K1=20, K2=50,K3=100, num_thread=1, device=device)
    HR1, NDCG1, map10,mrr10 = np.array(hits10).mean(), np.array(ndcgs10).mean(),np.array(maps10).mean(), np.array(mrrs10).mean()
    robust_HR10 = evaluate_adv.evalulate_robustness(args, model1, train_loader, optimizer1, args.epsilon, testRatings, testNegatives, 10, 20, 50, 100)
    logger.info("==> Finetune hits10: {:.4f}%, ndcg10: {:.4f}, robust HR10 : {:.4f}".format(HR1, NDCG1, robust_HR10))
    
    record = interpolation(args, logger, init_sd, model_copy, model1, train_loader, loss_function, save_dir)
    logger.info(record)
        
    modelPath = save_dir + f"/{args.data_set}_lr-{args.lr}_g-{args.awp_gamma}_FT_Inter.pth"
    os.makedirs("pretrained", exist_ok=True)
    torch.save(model1.state_dict(), modelPath)
    
import argparse
import logging
import sys
import time
import math
from bunch import Bunch
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from Dataset import Dataset
# from utils import DataLoader

from scipy.sparse import dok_matrix
from evaluate_adv import *
from utils_awp import AdvWeightPerturb1

upper_limit, lower_limit = 1,0

EPS = 1E-20

def cal_lp_norm(tensor,p,dim_count):
    tmp = tensor
    for i in range(1,dim_count):
        tmp = torch.norm(tmp,p=p,dim=i,keepdim=True) #torch.Size([100, 1])
    
    
    return torch.clamp_min(tmp, 1e-8)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='NeuMF') # NeuMF ConvNCF
    parser.add_argument('--l2', default=0, type=float)
    parser.add_argument('--l1', default=0, type=float)
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument("--data_path", nargs="?", default="Data/",
                        help="Input data path.")
    parser.add_argument("--dataset", nargs="?", default="ml-1m",   # ml-1m yelp  AToy lastfm
                        help="Choose a dataset.")
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr-max', default=0.0005, type=float)        # 0.1 这块应该是推荐模型训练学习率 RAT ml1m 0.0005
    parser.add_argument('--lr-prox', default=0.05, type=float)
    parser.add_argument('--attack', default='fgsm', type=str, choices=['fgsm', 'fgsm', 'bim', 'none' ,'Gaussian ','Uniform ', 'pgd', 'mim'])
    parser.add_argument('--epsilon', default=0.008 ,type=float) 
    parser.add_argument("--alpha", type=float ,default=0.008/10)  # 0.5/25  nargs="?",
    parser.add_argument("--num_steps", type=int, default=10)  #25
    # parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'])
    parser.add_argument('--fname', default='mlp_model', type=str)
    # parser.add_argument('--seed', default=123, type=int)
    # parser.add_argument('--half', action='store_true')
    # parser.add_argument('--width-factor', default=10, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--val', action='store_true')
    parser.add_argument("--decay_factor", nargs="?", default=1.0)
    # parser.add_argument('--chkpt-iters', default=10, type=int)
    # parser.add_argument('--awp-gamma', default=0.00001, type=float)
    # parser.add_argument('--awp-warmup', default=0, type=int)
    parser.add_argument("--fcLayers", nargs="?", default="[1024, 512, 256, 128, 64, 32, 16]", #  [512, 256, 128, 64, 32, 16]  [512,  128, 32]
                        help="Size of each layer. Note that the first layer is the "
                             "concatenation of user and item embeddings. So fcLayers[0]/2 is the embedding size.")
    parser.add_argument("--nNeg", type=int, default=4,help="Number of negative instances to pair with a positive instance.")
    parser.add_argument("--model_path", type=str, default="pretrained/ml-1m_MLP_TWP_23_1e-06_0.001_BEST.pth",help="pretrained/ml-1m_MLP_TWP_23_1e-06_0.001_BEST.pth")
 
    return parser.parse_args()

args = get_args()

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
        self.criterion = torch.nn.BCELoss()  # 添加损失函数
        
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

        # MLP_modules = []
        # self.num_layers = len(hidden_layer_MLP)
        # for i in range(self.num_layers):
        #     MLP_modules.append(nn.Dropout(p=self.dropout))
        #     if i == 0:
        #         MLP_modules.append(nn.Linear(embedding_dim_MLP*2, hidden_layer_MLP[0]))
        #     else:
        #         MLP_modules.append(nn.Linear(hidden_layer_MLP[i-1], hidden_layer_MLP[i]))
        #     MLP_modules.append(nn.ReLU())
        # self.MLP_layers = nn.Sequential(*MLP_modules)
        # self.predict_layer = nn.Linear(hidden_layer_MLP[-1]+embedding_dim_GMF, 1)

        if True:
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
        # 预测
        prediction = torch.sigmoid(self.predict_layer(concat)).view(-1)
        # 计算损失
        loss = self.criterion(prediction, label.float())
        
        # 返回损失和中间层输出
        return loss, *mlp_outputs, prediction


class ConvNCF(nn.Module):

    def __init__(self, user_count, item_count,device):
        super(ConvNCF, self).__init__()

        # some variables
        self.device = device
        self.user_count = user_count
        self.item_count = item_count
        # self.item_count = 12929 # AMusic
        # 添加损失函数
        self.criterion = torch.nn.BCEWithLogitsLoss()

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
        # self.fc = nn.Sequential(
        #     nn.Linear(32, 16),  # 输入维度改为64
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(16, 1)
        # )
                

    def forward(self, user_ids, item_ids, label):  # 添加label参数
        user_ids = list(map(int, user_ids))
        item_ids = list(map(int, item_ids))

        user_embeddings = self.P(torch.tensor(user_ids).to(self.device))
        item_embeddings = self.Q(torch.tensor(item_ids).to(self.device))

        interaction_map = torch.bmm(user_embeddings.unsqueeze(2), 
                                  item_embeddings.unsqueeze(1)).to(self.device)
        interaction_map = interaction_map.view((-1, 1, self.embedding_size, self.embedding_size))

        # 存储中间层输出
        intermediate_outputs = []
        x = interaction_map
        
        # 手动计算每个模块的输出
        for module in self.cnn:
            x = module(x)
            if isinstance(module, nn.ReLU):
                # 保存ReLU输出
                x.requires_grad_(True)
                x.retain_grad()
                intermediate_outputs.append(x)
        
        # 特征展平
        feature_vec = x.view((-1, 32))
        
        # 全连接层预测
        prediction = torch.sigmoid(self.fc(feature_vec).view(-1))
        prediction = torch.clamp(prediction, 1e-8, 1 - 1e-8)

        logits = self.fc(feature_vec).view(-1)

        if isinstance(label, bool):
            # 在评估模式下，我们不需要计算损失
            # 所以只需返回预测值
            return torch.sigmoid(logits)
        # 计算损失
        # label = label.to(prediction.device)
        # loss = self.criterion(prediction, label.float())

        label = label.to(logits.device)
        loss = self.criterion(logits, label.float())
        prediction = torch.sigmoid(logits)
        
        return loss, *intermediate_outputs, prediction
    

import torch
import zipfile
import io
import pickle
import os
def robust_load(path, device='cpu'):
    """
    健壮的模型加载函数，支持多种加载方式和恢复机制
    自动处理常见的模型保存格式
    """
    # 尝试加载整个文件
    loaded_obj = None
    error = None
    
    # 方法1: 尝试标准加载
    try:
        loaded_obj = torch.load(path, map_location=device, weights_only=False)
        print("✅ 标准加载成功")
    except Exception as e:
        error = f"标准加载失败: {str(e)}"
    
    # 方法2: 如果标准加载失败，尝试内部加载器
    if loaded_obj is None:
        try:
            with open(path, 'rb') as f:
                # 提供完整的参数
                loaded_obj = torch.serialization._load(
                    f, 
                    map_location=device,
                    pickle_module=pickle,
                    weights_only=False
                )
            print("✅ 内部加载器成功")
        except Exception as e:
            error = error or f"内部加载失败: {str(e)}"
    
    # 方法3: 手动处理ZIP内容（如果前两种失败）
    if loaded_obj is None:
        try:
            with zipfile.ZipFile(path, 'r') as zf:
                for name in zf.namelist():
                    if 'data.pkl' in name or name.endswith('.pkl'):
                        with zf.open(name) as f:
                            data = f.read()
                        loaded_obj = torch.load(io.BytesIO(data), map_location=device, weights_only=False)
                        break
            if loaded_obj is not None:
                print("✅ ZIP处理成功")
        except Exception as e:
            error = error or f"ZIP处理失败: {str(e)}"
    
    # 方法4: 暴力解析整个文件（如果前三种失败）
    if loaded_obj is None:
        try:
            with open(path, 'rb') as f:
                data = f.read()
                # 尝试直接反序列化
                try:
                    loaded_obj = torch.load(io.BytesIO(data), map_location=device, weights_only=False)
                except Exception as e:
                    pass
                
                # 尝试pickle加载
                if loaded_obj is None:
                    try:
                        loaded_obj = pickle.loads(data)
                    except Exception as e:
                        pass
                    
            if loaded_obj is not None:
                print("✅ 暴力解析成功")
        except Exception as e:
            error = error or f"暴力解析失败: {str(e)}"
    
    # 如果所有方法都失败
    if loaded_obj is None:
        raise RuntimeError(f"无法加载模型文件: {path}\n{error}")
    
    # ================= 关键修改：智能提取状态字典 =================
    # 1. 直接处理状态字典
    if isinstance(loaded_obj, (dict, collections.OrderedDict)):
        # 检查是否是常见的封装格式
        if 'state_dict' in loaded_obj:
            print("🔍 检测到'state_dict'键，提取模型参数")
            return loaded_obj['state_dict']
        
        # 检查是否是您的自定义格式
        if 'model' in loaded_obj:
            print("🔍 检测到'model'键，提取模型参数")
            return loaded_obj['model']
        
        # 检查是否有模块前缀的键
        has_module_prefix = any('module.' in key for key in loaded_obj.keys())
        if has_module_prefix:
            print("🔍 检测到包含'module.'前缀的键，作为状态字典处理")
            return loaded_obj
    
    # 2. 处理DataParallel封装的模型
    elif hasattr(loaded_obj, 'state_dict'):
        print("🔍 检测到包含state_dict方法的对象，使用该方法")
        return loaded_obj.state_dict()
    
    # 3. 检查直接状态字典特征
    elif isinstance(loaded_obj, torch.nn.Module):
        print("🔍 检测到nn.Module对象，直接返回其状态字典")
        return loaded_obj.state_dict()
    
    # 4. 非标准格式但可能是状态字典
    print("⚠️ 无法识别模型结构，尝试直接使用加载结果")
    return loaded_obj

def main():
    All_start_time = time.time()
    args = get_args()
    fcLayers = eval(args.fcLayers)
    topK = 10
    topK1 = 20
    topK2 = 50
    topK3 = 100
    print("比例l2,每一轮改变模型的参数都不变,不累积影响下去")
    print('--'*50)


    logger = logging.getLogger(__name__)    # 这段代码的作用是配置一个具有指定格式和级别的日志记录器，以便记录程序执行时的相关信息。文件和控制台都被用作输出目标。
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.fname, 'eval.log' if args.eval else 'output.log')),
            logging.StreamHandler()
        ])

    logger.info(args)
    
    # Loading data
    t1 = time.time()
    dataset = Dataset(args.data_path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    nUsers, nItems = train.shape
    
    arr = np.array(testRatings)
    max_value_second_column1 = np.max(arr[:, 1])
    arr = np.array(testNegatives)
    max_value_second_column2 = np.max(arr)
    n_value = max(max_value_second_column1, max_value_second_column2)
    if nItems <= n_value:
        nItems = n_value+1
        extended_sparse_matrix = dok_matrix((nUsers, nItems), dtype=np.float32)

        for (row, col), value in train.items():
            extended_sparse_matrix[row, col] = value
            train = extended_sparse_matrix
    
    userMatrix = torch.Tensor(get_train_matrix(train))

    itemMatrix = torch.transpose(torch.Tensor(get_train_matrix(train)), 0, 1)
    userMatrix, itemMatrix = userMatrix.to(device), itemMatrix.to(device)
    print(f"Load data: #user={nUsers}, #item={nItems}, #train={train.nnz}, #test={len(testRatings)} [{time.time()-t1:.1f}s]")
    print("MLP_Attack_test:*************************************")

    print(args.model)
    if args.model == "NeuMF":
        GMF_model = None
        MLP_model = None
        model_adv = NeuMF(nUsers, nItems, 32, 32, [64,32,16,8], 0.0, device, GMF_model,MLP_model)
    elif args.model == "ConvNCF":
        model_adv = ConvNCF(nUsers, nItems, device=device)
    else:
        raise ValueError("Unknown model")

    model_adv = nn.DataParallel(model_adv).to(device)

    if args.l2:
        decay, no_decay = [], []
        for name,param in model_adv.named_parameters():
            if 'bn' not in name and 'bias' not in name:
                decay.append(param)
            else:
                no_decay.append(param)
        params = [{'params':decay, 'weight_decay':args.l2},
                  {'params':no_decay, 'weight_decay': 0 }]
    else:
        params = model_adv.parameters()

    model_path = args.model_path

    print(model_path)
    print("attack_weight")
    # if 'robust'or 'MLP' in model_path: 
    if 'robust' in model_path:  
        # 假设 original_state_dict 是你加载的原始状态字典 
        original_state_dict = robust_load(model_path, device)
        # original_state_dict = torch.load(model_path, map_location=device, weights_only=False)
        # # 如果是mlp生成的模型，有net需要提取出来
        # original_state_dict = original_state_dict['net']
        
        # 修改键以匹配 DataParallel 的期望格式
        modified_state_dict = {'module.' + key: value for key, value in original_state_dict.items()}

        # 创建一个新的state_dict，只包含不带有'reg'字段的参数
        new_state_dict111 = {}
        for key, value in modified_state_dict.items():
            if 'reg' not in key:
                new_state_dict111[key] = value
        model_adv.load_state_dict(new_state_dict111)
        
        # # 现在使用修改后的状态字典加载模型
        # model_adv.load_state_dict(modified_state_dict) 
    else:  # RAWP
        original_state_dict = robust_load(model_path, device)
        # original_state_dict = torch.load(model_path, map_location=device, weights_only=False)
        # 创建一个新的state_dict，只包含不带有'reg'字段的参数
        modified_state_dict = {'module.' + key: value for key, value in original_state_dict.items()}
        original_state_dict = modified_state_dict

        new_state_dict111 = {}
        for key, value in original_state_dict.items():
            if 'reg' not in key:
                new_state_dict111[key] = value
        model_adv.load_state_dict(new_state_dict111)
        
    del new_state_dict111
    torch.cuda.empty_cache()

    # Check  performance
    t1 = time.time()
    hits10, ndcgs10, maps10, mrrs10 = evaluate_model(model_adv, testRatings, testNegatives, topK,topK1,topK2,topK3,args.attack,args.epsilon,args.alpha,args.num_steps, args.decay_factor, num_thread=1, device=device)
    hr10, ndcg10,map10,mrr10 = np.array(hits10).mean(), np.array(ndcgs10).mean(),np.array(maps10).mean(), np.array(mrrs10).mean()
    print(f"Init: HR10={hr10:.4f}, NDCG10={ndcg10:.4f}, mrrs10={mrr10:.4f}")
    

if __name__ == "__main__":
    main()

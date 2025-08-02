import os
import sys

from Dataset import Dataset


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
from scipy.sparse import dok_matrix
import data_util
import evaluate
import evaluate2
# import GMF
# import MLP

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

parser.add_argument("--data_path", type=str, default="Data/")
parser.add_argument("--model_path", type=str, default="/results/model")
parser.add_argument("--out", default=False, help="save model or not")
# parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
parser.add_argument("--enable_lat", nargs="?", default=True)
parser.add_argument("--alpha", nargs="?", default=1)
parser.add_argument("--pro_num", nargs="?", default=1, choices=[1, 25], help="1 for fgsm and 10 for bim/pgd")
parser.add_argument("--decay_factor", nargs="?", default=1.0)
parser.add_argument("--layerlist", nargs="?", default="all")
parser.add_argument("--adv", nargs="?", default=True)
parser.add_argument("--adv_reg", nargs="?", default=1)
parser.add_argument("--reg", nargs="?", default=1e-3)
parser.add_argument("--adv_type", nargs="?", default="fgsm", choices=['fgsm', 'bim', 'pgd','mim'])
parser.add_argument("--norm", nargs="?", default="linf", choices=['linf', 'l2'])
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
    plt.savefig("/data/chenhai-fwxz/DeepCF-PyTorch/result/%s_%s_%s.jpg" %(adv,title1, title2))

# class NeuMF(nn.Module):
#     def __init__(self, user_num, item_num, embedding_dim_MLP, hidden_layer_MLP,
#                  dropout,device, MLP_model=None):
#         super(NeuMF, self).__init__()
#         """
#         user_num: number of users;
#         item_num: number of items;
#         embedding_dim_GMF: number of embedding dimensions in GMF submodel;
#         embedding_dim_MLP: number of embedding dimensions in MLP submodel;
#         hidden_layer_MLP: dimension of each hidden layer (list type) in MLP submodel;
#         dropout: dropout rate between fully connected layers;
#         GMF_model: pre-trained GMF weights;
# 		MLP_model: pre-trained MLP weights.
#         """
#         self.device = device
#         self.dropout = dropout
#         # self.GMF_model = GMF_model
#         self.MLP_model = MLP_model

#         # self.embed_user_GMF = nn.Embedding(user_num, embedding_dim_GMF).to(self.device)
#         # self.embed_item_GMF = nn.Embedding(item_num, embedding_dim_GMF).to(self.device)
#         self.embed_user_MLP = nn.Embedding(user_num, embedding_dim_MLP).to(self.device)
#         self.embed_item_MLP = nn.Embedding(item_num, embedding_dim_MLP).to(self.device)


#         # MLP_modules = []
#         # self.num_layers = len(hidden_layer_MLP)
#         # for i in range(self.num_layers):
#         #     MLP_modules.append(nn.Dropout(p=self.dropout))
#         #     if i == 0:
#         #         MLP_modules.append(nn.Linear(embedding_dim_MLP*2, hidden_layer_MLP[0]))
#         #     else:
#         #         MLP_modules.append(nn.Linear(hidden_layer_MLP[i-1], hidden_layer_MLP[i]))
#         #     MLP_modules.append(nn.ReLU())
#         # self.MLP_layers = nn.Sequential(*MLP_modules)

#         MLP_modules = []
#         for l1, l2 in zip(hidden_layer_MLP[:-1], hidden_layer_MLP[1:]):
#             MLP_modules.append(nn.Linear(l1, l2))
#             MLP_modules.append(nn.ReLU(inplace=True))
#         self.MLP_layers = nn.Sequential(*MLP_modules)

#         self.predict_layer = nn.Linear(hidden_layer_MLP[-1], 1)

#         if not args.use_pretrained:
#             # nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
#             # nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
#             nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
#             nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

#             for m in self.MLP_layers:
#                 if isinstance(m, nn.Linear):
#                     nn.init.xavier_uniform_(m.weight)
#             nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')


#             # Kaiming/Xavier initialization can not deal with non-zero bias terms
#             for m in self.modules():
#                 if isinstance(m, nn.Linear) and m.bias is not None:
#                     m.bias.data.zero_()
#         else:
#             # embedding layers
#             # self.embed_user_GMF.weight.data.copy_(self.GMF_model.embed_user.weight)
#             # self.embed_item_GMF.weight.data.copy_(self.GMF_model.embed_item.weight)
#             self.embed_user_MLP.weight.data.copy_(self.MLP_model.embed_user.weight)
#             self.embed_item_MLP.weight.data.copy_(self.MLP_model.embed_item.weight)

#             # mlp layers
#             for (m1, m2) in zip(self.MLP_layers, self.MLP_model.MLP_layers):
#                 if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
#                     m1.weight.data.copy_(m2.weight)
#                     m1.bias.data.copy_(m2.bias)

#             # predict layers
#             predict_weight =  self.MLP_model.predict_layer.weight
#             precit_bias = self.MLP_model.predict_layer.bias

#             self.predict_layer.weight.data.copy_(0.5 * predict_weight)
#             self.predict_layer.bias.data.copy_(0.5 * precit_bias)

#     def forward(self, user, item):
#         # embed_user_GMF = self.embed_user_GMF(user).to(self.device)
#         # embed_item_GMF = self.embed_item_GMF(item).to(self.device)
#         # output_GMF = embed_user_GMF * embed_item_GMF

#         embed_user_MLP = self.embed_user_MLP(user).to(self.device)
#         embed_item_MLP = self.embed_item_MLP(item).to(self.device)
#         interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1).to(self.device)
#         output_MLP = self.MLP_layers(interaction).to(self.device)

#         # concat = torch.cat((output_GMF, output_MLP), -1).to(self.device)

#         prediction = self.predict_layer(output_MLP)
#         return prediction.view(-1)




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
        
        # embedding_dim_MLP = tuple(embedding_dim_MLP)
        self.embed_user_GMF = nn.Embedding(user_num, embedding_dim_GMF).to(self.device)
        self.embed_item_GMF = nn.Embedding(item_num, embedding_dim_GMF).to(self.device)
        self.embed_user_MLP = nn.Embedding(user_num, embedding_dim_MLP).to(self.device)
        self.embed_item_MLP = nn.Embedding(item_num, embedding_dim_MLP).to(self.device)


        MLP_modules = []
        self.num_layers = len(hidden_layer_MLP)
        for i in range(self.num_layers):
            MLP_modules.append(nn.Dropout(p=self.dropout))
            if i == 0:
                MLP_modules.append(nn.Linear(embedding_dim_MLP*2, hidden_layer_MLP[0]))
            else:
                MLP_modules.append(nn.Linear(hidden_layer_MLP[i-1], hidden_layer_MLP[i]))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        # MLP_modules = []
        # for l1, l2 in zip(hidden_layer_MLP[:-1], hidden_layer_MLP[1:]):
        #     MLP_modules.append(nn.Linear(l1, l2))
        #     MLP_modules.append(nn.ReLU(inplace=True))
        # self.MLP_layers = nn.Sequential(*MLP_modules)

        self.predict_layer = nn.Linear(hidden_layer_MLP[-1]+embedding_dim_GMF, 1)

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

    def forward(self, user, item):
        embed_user_GMF = self.embed_user_GMF(user).to(self.device)
        embed_item_GMF = self.embed_item_GMF(item).to(self.device)
        output_GMF = embed_user_GMF * embed_item_GMF

        embed_user_MLP = self.embed_user_MLP(user).to(self.device)
        embed_item_MLP = self.embed_item_MLP(item).to(self.device)
        interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1).to(self.device)
        output_MLP = self.MLP_layers(interaction).to(self.device)

        concat = torch.cat((output_GMF, output_MLP), -1).to(self.device)

        # prediction = self.predict_layer(output_MLP)
        
        prediction = self.predict_layer(concat)
        return prediction.view(-1)


if __name__=="__main__":
    print(f"mine_lr={args.lr}, dataset:{args.data_set}")
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
    model1 = NeuMF(user_num, item_num,args.embedding_dim_GMF, args.embedding_dim_MLP, args.hidden_layer_MLP, args.dropout, device,GMF_model, MLP_model)
    model1.to(device)
    # RAWP_model2 = NeuMF(user_num, item_num, args.embedding_dim_MLP, args.hidden_layer_MLP, args.dropout,args.epsilon, args.pro_num,\
    #             args.enable_lat, args.layerlist, args.adv_type, args.adv_reg, args.norm,args.decay_factor, device, MLP_model)
    # model2.to(device)
    loss_function = nn.BCEWithLogitsLoss()

    if args.use_pretrained:
        optimizer1 = optim.SGD(model1.parameters(), lr=args.lr)
        # optimizer2 = optim.SGD(model2.parameters(), lr=args.lr)
    else:
        optimizer1 = optim.Adam(model1.parameters(), lr=args.lr)
        # optimizer2 = optim.Adam(model2.parameters(), lr=args.lr)

    # writer = SummaryWriter() # for visualization

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
        print("NeuMF: The time elapse of epoch {:03d}".format(epoch1) + " is: " +
                time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        print("HR: {:.4f}\tNDCG: {:.4f}".format(np.mean(HR1), np.mean(NDCG1)))
        modelPath = f"pretrained/{args.data_set}_NeuMF_adv.pth"
        if HR1 > best_hr1:
            best_hr1, best_ndcg1, best_epoch1 = HR1, NDCG1, epoch1
            os.makedirs("pretrained", exist_ok=True)
            torch.save(model1.state_dict(), modelPath)
            print("save:=================================")
            print(modelPath)
        
    # for epoch2 in range(args.epochs):   
    #     if args.enable_lat:
    #         model2.update_seed() 

    #     if args.enable_lat:
    #         args.epsilon, args.pro_num = set_noise(epoch2,adv_type=args.adv_type)
    #         model2.update_noise(args.epsilon, args.pro_num)
        
    #     for user, item, label in train_loader:
    #         user = user.to(device)
    #         item = item.to(device)
    #         label = label.float().to(device)
    #         prediction2,loss2 = model2(user, item,label)

    #         optimizer2.zero_grad()
    #         loss2.backward()
    #         optimizer2.step()
    #         if args.enable_lat:
    #             model2.save_grad()
    #         # writer.add_scalar('data/loss', loss.item(), count)
    #         count2 += 1
    #     HR2, NDCG2 = evaluate2.metrics(model2, test_loader, args.top_k, device) 
    #     advHR10.append(np.mean(HR2))
    #     advNDCG10.append(np.mean(NDCG2))
    #     elapsed_time = time.time() - start_time
    #     print("model2: The time elapse of epoch {:03d}".format(epoch2) + " is: " +
    #             time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
    #     print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR2), np.mean(NDCG2)))
    #     if HR2 > best_hr2:
    #         best_hr2, best_ndcg2, best_epoch2 = HR2, NDCG2, epoch2

    # show_metric(args.epochs, HR10,NDCG10,advHR10,advNDCG10, 'HR@10', 'NDCG@10', args.adv_type)
    # # show_curve(20, NDCG10, 'NDCG@10') 

    print("End model1. Best epoch {:03d}: HR = {:.4f}, NDCG = {:.4f}".format(best_epoch1, best_hr1, best_ndcg1))
    # print("End model2. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(best_epoch2, best_hr2, best_ndcg2))
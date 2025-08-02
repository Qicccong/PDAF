import argparse
import os
import time

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# class MLP(nn.Module):
#     def __init__(self, fc_layers, user_matrix, item_matrix,device):
#         super(MLP, self).__init__()

#         # Define layers
#         self.device = device
#         self.user_fc = nn.Linear(item_matrix.size(0), fc_layers[0]//2)
#         self.item_fc = nn.Linear(user_matrix.size(0), fc_layers[0]//2)
#         layers = []
#         for l1, l2 in zip(fc_layers[:-1], fc_layers[1:]):
#             layers += [nn.Linear(l1, l2), nn.ReLU(inplace=True)]
#         self.fcs = nn.Sequential(*layers)
#         self.final = nn.Sequential(nn.Linear(fc_layers[-1], 1), nn.Sigmoid())

#         # Register user and item matrices as buffers
#         self.register_buffer("user_matrix", user_matrix)
#         self.register_buffer("item_matrix", item_matrix)
#         print(self)

#     def forward(self, user, item):
#         userInput = self.user_matrix[user, :].to(self.device)          # (B, 3706)
#         itemInput = self.item_matrix[item, :].to(self.device)          # (B, 6040)
#         user_vector = self.user_fc(userInput).to(self.device)           # (B, fcLayers[0]//2)
#         item_vector = self.item_fc(itemInput).to(self.device)           # (B, fcLayers[0]//2)


#         # Concatenate user and item vectors
#         x = torch.cat((user_vector, item_vector), -1)

#         # Pass through fully connected layers
#         x = self.fcs(x)

#         # Final layer
#         y = self.final(x)

#         return y.squeeze()


# class MLP(nn.Module):
#     def __init__(self, fcLayers, userMatrix, itemMatrix,device):
#         super(MLP, self).__init__()
#         self.criterion = torch.nn.BCELoss()
#         self.device = device
#         self.seed1,_,_,_,_,_ = self.random()
#         _,self.seed2,_,_,_,_ = self.random()
#         _,_,self.seed3,_,_,_ = self.random()
#         _,_,_,self.seed4,_,_ = self.random()
#         _,_,_,_,self.seed5,_ = self.random()
#         _,_,_,_,_,self.seed6 = self.random()
#         self.maxseed = max(self.seed1, self.seed2,self.seed3,self.seed4,self.seed5,self.seed6)
#         self.y_list = ['y0_reg','y1_reg','y2_reg','y3_reg','y4_reg','y5_reg','y6_reg']
#         self.enable_list = [0 for i in range(7)]
#         self.layerlist = "all"
#         self.layerlist_digit = list(range(0,self.maxseed+1))
#         self.register_buffer("userMatrix", userMatrix)
#         self.register_buffer("itemMatrix", itemMatrix)
        
#         nUsers = self.userMatrix.size(0)
#         nItems = self.itemMatrix.size(0)


#         # In the official implementation, 
#         # the first dense layer has no activation
#         self.userFC = nn.Linear(nItems, fcLayers[0]//2)
#         self.itemFC = nn.Linear(nUsers, fcLayers[0]//2)
#         self.reg_size_list = list()
#         layers = []
#         self.register_buffer(self.y_list[0], torch.zeros([100, 1024]))
#         y_index = 1
#         for l1, l2 in zip(fcLayers[:-1], fcLayers[1:]):
#             layers+=[nn.Linear(l1, l2)]
#             layers+=[nn.ReLU(inplace=True)]
#             self.register_buffer(self.y_list[y_index], torch.zeros([100, l2]))
#             self.reg_size_list.append([100, l2])
#             y_index+=1
#         self.fcs = nn.Sequential(*layers)

#         # In the official implementation, 
#         # the final module is initialized using Lecun normal method.
#         # Here, the Kaiming normal initialization is adopted.
#         self.final = nn.Sequential(
#             nn.Linear(fcLayers[-1], 1),
#             nn.Sigmoid(),
#         )
#         # self.choose_layer()
#         print(self)

#     def forward(self, user, item,label):
#         userInput = self.userMatrix[user, :].to(self.device)         # (B, 3706)
#         itemInput = self.itemMatrix[item, :].to(self.device)         # (B, 6040)
#         userVector = self.userFC(userInput).to(self.device)          # (B, fcLayers[0]//2)
#         itemVector = self.itemFC(itemInput).to(self.device)           # (B, fcLayers[0]//2)
#         self.y0 = torch.cat((userVector, itemVector), -1).to(self.device)   # (B, fcLayers[0])
        
#         self.y1 = self.fcs[0](self.y0).to(self.device)
#         self.y1 = self.fcs[1](self.y1) 
        
#         self.y2 = self.fcs[2](self.y1).to(self.device)
#         self.y2 = self.fcs[3](self.y2)
        
#         self.y3 = self.fcs[4](self.y2).to(self.device)
#         self.y3 = self.fcs[5](self.y3)

#         self.y4 = self.fcs[6](self.y3).to(self.device)
#         self.y4 = self.fcs[7](self.y4)

#         self.y5 = self.fcs[8](self.y4).to(self.device)
#         self.y5 = self.fcs[9](self.y5)

#         self.y6 = self.fcs[10](self.y5).to(self.device)
#         self.y6 = self.fcs[11](self.y6)
#                                      # (B, fcLayers[-1])
#         self.y = self.final(self.y6).to(self.device)
#         self.yc = self.y.squeeze()
#         loss = self.criterion(self.yc, label)
        
#         self.y1_add=0
#         self.y2_add=0
#         self.y3_add=0
#         self.y4_add=0
#         self.y5_add=0
#         self.y6_add=0

       
#         return loss, self.y1_add, self.y2_add,self.y3_add,self.y4_add,self.y5_add,self.y6_add, self.y
#         # return loss,0,0,0,0,0,0,self.y


class MLP(nn.Module):
    def __init__(self, fcLayers, userMatrix, itemMatrix,device):
        super(MLP, self).__init__()
        self.device = device
        self.register_buffer("userMatrix", userMatrix)
        self.register_buffer("itemMatrix", itemMatrix)
        nUsers = self.userMatrix.size(0)
        nItems = self.itemMatrix.size(0)
        self.criterion = torch.nn.BCELoss()
        
        # In the official implementation, 
        # the first dense layer has no activation
        self.userFC = nn.Linear(nItems, fcLayers[0]//2)
        self.itemFC = nn.Linear(nUsers, fcLayers[0]//2)
        layers = []
        for l1, l2 in zip(fcLayers[:-1], fcLayers[1:]):
            layers.append(nn.Linear(l1, l2))
            layers.append(nn.ReLU(inplace=False))
        self.fcs = nn.Sequential(*layers)

        # In the official implementation, 
        # the final module is initialized using Lecun normal method.
        # Here, the Kaiming normal initialization is adopted.
        self.final = nn.Sequential(
            nn.Linear(fcLayers[-1], 1),
            nn.Sigmoid(),
        )

    def forward(self, user, item, label ):
        userInput = self.userMatrix[user, :].to(self.device)        # (B, 3706)
        itemInput = self.itemMatrix[item, :].to(self.device)          # (B, 6040)
        userVector = self.userFC(userInput).to(self.device)           # (B, fcLayers[0]//2)
        itemVector = self.itemFC(itemInput).to(self.device)          # (B, fcLayers[0]//2)
        self.y0 = torch.cat((userVector, itemVector), -1).to(self.device) 
        self.y1 = self.fcs[0](self.y0).to(self.device)
        self.y1.requires_grad_(True)   #241029 跑干净MLP的时候，这里总报错，所以暂时注释掉
        self.y1.retain_grad()
        self.y2 = self.fcs[1](self.y1) 
        
            
        # # 获取 self.fcs[1] 中每个子层的参数
        # for idx, layer in enumerate(self.fcs.children()):
        #     print(f"Layer {idx}:")
        #     for name, param in layer.named_parameters():
        #         print(f"Parameter name: {name}, Value: {param}")
        
        self.y3 = self.fcs[2](self.y2).to(self.device)
        self.y3.requires_grad_(True)
        self.y3.retain_grad()
        self.y4 = self.fcs[3](self.y3)
        
        self.y5 = self.fcs[4](self.y4).to(self.device)
        self.y5.requires_grad_(True)
        self.y5.retain_grad()
        self.y6 = self.fcs[5](self.y5)

        self.y7 = self.fcs[6](self.y6).to(self.device)
        self.y7.requires_grad_(True)
        self.y7.retain_grad()
        self.y8 = self.fcs[7](self.y7)

        self.y9 = self.fcs[8](self.y8).to(self.device)
        self.y9.requires_grad_(True)
        self.y9.retain_grad()
        self.y10 = self.fcs[9](self.y9)

        self.y11 = self.fcs[10](self.y10).to(self.device)
        self.y11.requires_grad_(True)
        self.y11.retain_grad()
        self.y12 = self.fcs[11](self.y11)
                                     # (B, fcLayers[-1])
        y = self.final(self.y12).to(self.device)
                              # (B, fcLayers[-1])                          # (B, 1)
        yc = y.squeeze()
        loss = self.criterion(yc, label)
        return loss,self.y1,self.y3,self.y5,self.y7,self.y9,self.y11, y


    def choose_layer(self):
        # if self.enable_lat == False:
        #     return
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


class MLP_4(nn.Module):
    def __init__(self, fcLayers, userMatrix, itemMatrix,device):
        super(MLP_4, self).__init__()
        self.criterion = torch.nn.BCELoss()
        self.device = device
        self.seed1,_,_,_ = self.random()
        _,self.seed2,_,_ = self.random()
        _,_,self.seed3,_ = self.random()
        _,_,_,self.seed4 = self.random()
        self.maxseed = max(self.seed1, self.seed2,self.seed3,self.seed4)
        self.y_list = ['y0_reg','y1_reg','y2_reg','y3_reg','y4_reg']
        self.enable_list = [0 for i in range(5)]
        self.layerlist = "all"
        self.layerlist_digit = list(range(0,self.maxseed+1))
        self.register_buffer("userMatrix", userMatrix)
        self.register_buffer("itemMatrix", itemMatrix)
        
        nUsers = self.userMatrix.size(0)
        nItems = self.itemMatrix.size(0)


        # In the official implementation, 
        # the first dense layer has no activation
        self.userFC = nn.Linear(nItems, fcLayers[0]//2)
        self.itemFC = nn.Linear(nUsers, fcLayers[0]//2)
        self.reg_size_list = list()
        layers = []
        self.register_buffer(self.y_list[0], torch.zeros([100, 1024]))
        y_index = 1
        for l1, l2 in zip(fcLayers[:-1], fcLayers[1:]):
            layers+=[nn.Linear(l1, l2)]
            layers+=[nn.ReLU(inplace=True)]
            self.register_buffer(self.y_list[y_index], torch.zeros([100, l2]))
            self.reg_size_list.append([100, l2])
            y_index+=1
        self.fcs = nn.Sequential(*layers)

        # In the official implementation, 
        # the final module is initialized using Lecun normal method.
        # Here, the Kaiming normal initialization is adopted.
        self.final = nn.Sequential(
            nn.Linear(fcLayers[-1], 1),
            nn.Sigmoid(),
        )
        self.choose_layer()
        print(self)

    def forward(self, user, item,label):
        userInput = self.userMatrix[user, :].to(self.device)         # (B, 3706)
        itemInput = self.itemMatrix[item, :].to(self.device)         # (B, 6040)
        userVector = self.userFC(userInput).to(self.device)          # (B, fcLayers[0]//2)
        itemVector = self.itemFC(itemInput).to(self.device)           # (B, fcLayers[0]//2)
        self.y0 = torch.cat((userVector, itemVector), -1).to(self.device)   # (B, fcLayers[0])
        
        self.y1 = self.fcs[0](self.y0).to(self.device)
        self.y1 = self.fcs[1](self.y1) 
        
        self.y2 = self.fcs[2](self.y1).to(self.device)
        self.y2 = self.fcs[3](self.y2)
        
        self.y3 = self.fcs[4](self.y2).to(self.device)
        self.y3 = self.fcs[5](self.y3)

        self.y4 = self.fcs[6](self.y3).to(self.device)
        self.y4 = self.fcs[7](self.y4)

                                     # (B, fcLayers[-1])
        self.y = self.final(self.y4).to(self.device)
        self.yc = self.y.squeeze()
        loss = self.criterion(self.yc, label)
       
        # return loss, self.y1_add, self.y2_add,self.y3_add,self.y4_add,self.y5_add,self.y6_add, self.y
        return loss,0,0,0,0,0,0,self.y

    def choose_layer(self):
        # if self.enable_lat == False:
        #     return
        if self.layerlist == 'all':
            self.enable_list1 = list(range(0, self.seed1+1))
            self.enable_list2 = list(range(0, self.seed2+1))
            self.enable_list3 = list(range(0, self.seed3+1))
            self.enable_list4 = list(range(0, self.seed4+1))
   # all True
        else:
            for i in self.layerlist_digit:
                self.enable_list[i] = 1


    def update_seed(self):
        self.seed1,self.seed2,self.seed3,self.seed4 = self.random()

    def random(self):
        seed = torch.rand(4)*0.7
        zs1= int(torch.clamp(seed[0]*10, min=0, max=4))
        zs2 = int(torch.clamp(seed[1]*10, min=0, max=4))
        zs3 = int(torch.clamp(seed[2]*10, min=0, max=4))
        zs4 = int(torch.clamp(seed[3]*10, min=0, max=4))
        return zs1,zs2,zs3,zs4

   

# class MLP(nn.Module):
#     def __init__(self, fcLayers, userMatrix, itemMatrix,device):
#         super(MLP, self).__init__()
#         self.criterion = torch.nn.BCELoss()
#         self.device = device
#         self.seed1,_,_,_,_,_ = self.random()
#         _,self.seed2,_,_,_,_ = self.random()
#         _,_,self.seed3,_,_,_ = self.random()
#         _,_,_,self.seed4,_,_ = self.random()
#         _,_,_,_,self.seed5,_ = self.random()
#         _,_,_,_,_,self.seed6 = self.random()
#         self.maxseed = max(self.seed1, self.seed2,self.seed3,self.seed4,self.seed5,self.seed6)
#         self.y_list = ['y0_reg','y1_reg','y2_reg','y3_reg','y4_reg','y5_reg','y6_reg']
#         self.enable_list = [0 for i in range(7)]
#         self.layerlist = "all"
#         self.layerlist_digit = list(range(0,self.maxseed+1))
#         self.register_buffer("userMatrix", userMatrix)
#         self.register_buffer("itemMatrix", itemMatrix)
        
#         nUsers = self.userMatrix.size(0)
#         nItems = self.itemMatrix.size(0)


#         # In the official implementation, 
#         # the first dense layer has no activation
#         self.userFC = nn.Linear(nItems, fcLayers[0]//2)
#         self.itemFC = nn.Linear(nUsers, fcLayers[0]//2)
#         self.reg_size_list = list()
#         layers = []
#         self.register_buffer(self.y_list[0], torch.zeros([100, 1024]))
#         y_index = 1
#         for l1, l2 in zip(fcLayers[:-1], fcLayers[1:]):
#             layers+=[nn.Linear(l1, l2)]
#             layers+=[nn.ReLU(inplace=True)]
#             self.register_buffer(self.y_list[y_index], torch.zeros([100, l2]))
#             self.reg_size_list.append([100, l2])
#             y_index+=1
#         self.fcs = nn.Sequential(*layers)

#         # In the official implementation, 
#         # the final module is initialized using Lecun normal method.
#         # Here, the Kaiming normal initialization is adopted.
#         self.final = nn.Sequential(
#             nn.Linear(fcLayers[-1], 1),
#             nn.Sigmoid(),
#         )
#         self.choose_layer()
#         print(self)

#     def forward(self, user, item):
#         userInput = self.userMatrix[user, :].to(self.device)         # (B, 3706)
#         itemInput = self.itemMatrix[item, :].to(self.device)         # (B, 6040)
#         userVector = self.userFC(userInput).to(self.device)          # (B, fcLayers[0]//2)
#         itemVector = self.itemFC(itemInput).to(self.device)           # (B, fcLayers[0]//2)
#         self.y0 = torch.cat((userVector, itemVector), -1).to(self.device)   # (B, fcLayers[0])
        
#         self.y1 = self.fcs[0](self.y0).to(self.device)
#         self.y1 = self.fcs[1](self.y1) 
        
#         self.y2 = self.fcs[2](self.y1).to(self.device)
#         self.y2 = self.fcs[3](self.y2)
        
#         self.y3 = self.fcs[4](self.y2).to(self.device)
#         self.y3 = self.fcs[5](self.y3)

#         self.y4 = self.fcs[6](self.y3).to(self.device)
#         self.y4 = self.fcs[7](self.y4)

#         self.y5 = self.fcs[8](self.y4).to(self.device)
#         self.y5 = self.fcs[9](self.y5)

#         self.y6 = self.fcs[10](self.y5).to(self.device)
#         self.y6 = self.fcs[11](self.y6)
#                                      # (B, fcLayers[-1])
#         self.y = self.final(self.y6).to(self.device)
#         self.yc = self.y.squeeze()
#         # loss = self.criterion(self.yc, label)
       
#         # return loss, self.y1_add, self.y2_add,self.y3_add,self.y4_add,self.y5_add,self.y6_add, self.y
#         return self.yc,self.y

#     def choose_layer(self):
#         # if self.enable_lat == False:
#         #     return
#         if self.layerlist == 'all':
#             self.enable_list1 = list(range(0, self.seed1+1))
#             self.enable_list2 = list(range(0, self.seed2+1))
#             self.enable_list3 = list(range(0, self.seed3+1))
#             self.enable_list4 = list(range(0, self.seed4+1))
#             self.enable_list5 = list(range(0, self.seed5+1))
#             self.enable_list6 = list(range(0, self.seed6+1))   # all True
#         else:
#             for i in self.layerlist_digit:
#                 self.enable_list[i] = 1


#     def update_seed(self):
#         self.seed1,self.seed2,self.seed3,self.seed4,self.seed5,self.seed6 = self.random()

#     def random(self):
#         seed = torch.rand(6)*0.7
#         zs1= int(torch.clamp(seed[0]*10, min=0, max=6))
#         zs2 = int(torch.clamp(seed[1]*10, min=0, max=6))
#         zs3 = int(torch.clamp(seed[2]*10, min=0, max=6))
#         zs4 = int(torch.clamp(seed[3]*10, min=0, max=6))
#         zs5 = int(torch.clamp(seed[4]*10, min=0, max=6))
#         zs6 = int(torch.clamp(seed[5]*10, min=0, max=6))
#         return zs1,zs2,zs3,zs4,zs5,zs6


class MLP_new1(nn.Module):
    def __init__(self, model, userMatrix, itemMatrix, device):
        super(MLP_new1, self).__init__()
        self.device = device
        self.model = model
        self.register_buffer("userMatrix", userMatrix)
        self.register_buffer("itemMatrix", itemMatrix)

    def forward(self, y, user,item, label):
        _,self.y1,self.y3,self.y5,self.y7,self.y9,self.y11, y = self.model(user,item, label)
        userInput = self.userMatrix[user, :].to(self.device)        # (B, 3706)
        itemInput = self.itemMatrix[item, :].to(self.device)          # (B, 6040)
        userVector = self.model.userFC(userInput).to(self.device)           # (B, fcLayers[0]//2)
        itemVector = self.model.itemFC(itemInput).to(self.device)          # (B, fcLayers[0]//2)
        self.y0 = torch.cat((userVector, itemVector), -1).to(self.device) 
        self.y1 = self.model.fcs[0](self.y0).to(self.device)
        if y.shape == self.y1.shape:
            self.y1=y
        self.y2 = self.model.fcs[1](self.y1) 
        
        self.y3 = self.model.fcs[2](self.y2).to(self.device)
        if y.shape == self.y3.shape:
            self.y3=y
        self.y4 = self.model.fcs[3](self.y3)
        
        self.y5 = self.model.fcs[4](self.y4).to(self.device)
        if y.shape == self.y5.shape:
            self.y5=y
        self.y6 = self.model.fcs[5](self.y5)

        self.y7 = self.model.fcs[6](self.y6).to(self.device)
        if y.shape == self.y7.shape:
            self.y7=y
        self.y8 = self.model.fcs[7](self.y7)

        self.y9 = self.model.fcs[8](self.y8).to(self.device)
        if y.shape == self.y9.shape:
            self.y9=y
        self.y10 = self.model.fcs[9](self.y9)

        self.y11 = self.model.fcs[10](self.y10).to(self.device)
        if y.shape == self.y11.shape:
            self.y11=y
        self.y12 = self.model.fcs[11](self.y11)
                                     # (B, fcLayers[-1])
        y = self.model.final(self.y12).to(self.device)
                              # (B, fcLayers[-1])                          # (B, 1)
        return self.y1,self.y3,self.y5,self.y7,self.y9,self.y11, y

class MLP_white_attack_test(nn.Module):
    def __init__(self, fcLayers, userMatrix, itemMatrix, alpha, epsilon, pro_num, enable_lat, layerlist, adv_type, adv_reg,norm,decay_factor, device):
        super(MLP_white_attack_test, self).__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.adv_type = adv_type
        self.norm = norm
        self.criterion = torch.nn.BCELoss()
        self.device = device
        self.adv_reg = adv_reg
        self.decay_factor = decay_factor
        self.pro_num = pro_num
        self.enable_lat = enable_lat
        # self.seed1,_,_,_,_,_ = self.random()
        # _,self.seed2,_,_,_,_ = self.random()
        # _,_,self.seed3,_,_,_ = self.random()
        # _,_,_,self.seed4,_,_ = self.random()
        # _,_,_,_,self.seed5,_ = self.random()
        # _,_,_,_,_,self.seed6 = self.random()
        # self.maxseed = max(self.seed1, self.seed2,self.seed3,self.seed4,self.seed5,self.seed6)
        self.y_list = ['y0_reg','y1_reg','y2_reg','y3_reg','y4_reg','y5_reg','y6_reg']
        self.enable_list = [0 for i in range(7)]
        if enable_lat and layerlist != "all":
            self.layerlist = [int(x) for x in layerlist.split(',')]
            self.layerlist_digit = [int(x) for x in layerlist.split(',')]
        else:
            self.layerlist = "all"
            self.layerlist_digit = list(range(0,self.maxseed+1))
        self.register_buffer("userMatrix", userMatrix)
        self.register_buffer("itemMatrix", itemMatrix)
        
        nUsers = self.userMatrix.size(0)
        nItems = self.itemMatrix.size(0)


        # In the official implementation, 
        # the first dense layer has no activation
        self.userFC = nn.Linear(nItems, fcLayers[0]//2)
        self.itemFC = nn.Linear(nUsers, fcLayers[0]//2)
        self.reg_size_list = list()
        layers = []
        self.register_buffer(self.y_list[0], torch.zeros([100, 1024]))
        y_index = 1
        for l1, l2 in zip(fcLayers[:-1], fcLayers[1:]):
            layers+=[nn.Linear(l1, l2)]
            layers+=[nn.ReLU(inplace=True)]
            self.register_buffer(self.y_list[y_index], torch.zeros([100, l2]))
            self.reg_size_list.append([100, l2])
            y_index+=1
        self.fcs = nn.Sequential(*layers)

        # In the official implementation, 
        # the final module is initialized using Lecun normal method.
        # Here, the Kaiming normal initialization is adopted.
        self.final = nn.Sequential(
            nn.Linear(fcLayers[-1], 1),
            nn.Sigmoid(),
        )
        self.choose_layer()

    def forward(self, user, item, label):
        userInput = self.userMatrix[user, :].to(self.device)         # (B, 3706)
        itemInput = self.itemMatrix[item, :].to(self.device)         # (B, 6040)
        userVector = self.userFC(userInput).to(self.device)          # (B, fcLayers[0]//2)
        itemVector = self.itemFC(itemInput).to(self.device)           # (B, fcLayers[0]//2)
        self.y0 = torch.cat((userVector, itemVector), -1).to(self.device)   # (B, fcLayers[0])
        
        self.y1 = self.fcs[0](self.y0).to(self.device)
        self.y1 = self.fcs[1](self.y1) 
        
        self.y2 = self.fcs[2](self.y1).to(self.device)
        self.y2 = self.fcs[3](self.y2)
        
        self.y3 = self.fcs[4](self.y2).to(self.device)
        self.y3 = self.fcs[5](self.y3)

        self.y4 = self.fcs[6](self.y3).to(self.device)
        self.y4 = self.fcs[7](self.y4)

        self.y5 = self.fcs[8](self.y4).to(self.device)
        self.y5 = self.fcs[9](self.y5)

        self.y6 = self.fcs[10](self.y5).to(self.device)
        self.y6 = self.fcs[11](self.y6)
                                     # (B, fcLayers[-1])
        self.y = self.final(self.y6).to(self.device)
        self.yc = self.y.squeeze()
        loss = self.criterion(self.yc, label)
        
        if self.enable_lat:
            self.y1 = self.fcs[0](self.y0).to(self.device)  #torch.Size([100, 100])         
            if self.enable_lat and 1 in self.enable_list1:
                self.y1.requires_grad_(True)
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
            self.z1 = self.fcs[1](self.y1_add).to(self.device) 
            
            self.y2 = self.fcs[2](self.z1).to(self.device) 
            if self.enable_lat and 2 in self.enable_list2:
                self.y2.requires_grad_(True)  # 3333333333333333333333333333333
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
            self.z2 = self.fcs[3](self.y2_add).to(self.device)  #torch.Size([100, 128])

            self.y3 = self.fcs[4](self.z2).to(self.device)   
            if self.enable_lat and 3 in self.enable_list3:
                self.y3.requires_grad_(True)   # 333333333333333333333333333333333333333333333333
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
            self.z3 = self.fcs[5](self.y3_add).to(self.device) 

            self.y4 = self.fcs[6](self.z3).to(self.device)  
            if self.enable_lat and 4 in self.enable_list4:
                self.y4.requires_grad_(True)   # 3333333333333333333333333333333333
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
            self.z4 = self.fcs[7](self.y4_add)

            self.y5 = self.fcs[8](self.z4).to(self.device)  
            if self.enable_lat and 5 in self.enable_list5:
                self.y5.requires_grad_(True)  # 33333333333333333333333333333333333333333333333
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
            self.z5 = self.fcs[9](self.y5_add)

            self.y6 = self.fcs[10](self.z5).to(self.device)  
            if self.enable_lat and 6 in self.enable_list6:
                self.y6.requires_grad_(True)  # 333333333333333333333333333333333333
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
            self.z6 = self.fcs[11](self.y6_add)

            # print(y3_add.data)                            
            self.y = self.final(self.z6).to(self.device)
            self.yc = self.y.squeeze()
            loss_adv = self.criterion(self.yc, label)
            loss = loss + self.adv_reg * loss_adv                           # (B, 1)
        return loss, self.y1_add, self.y2_add,self.y3_add,self.y4_add,self.y5_add,self.y6_add, self.y

    

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

def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP.")
    parser.add_argument("--path", nargs="?", default="Data/",
                        help="Input data path.")
    parser.add_argument("--dataset", nargs="?", default="yelp",
                        help="Choose a dataset.")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs.")
    parser.add_argument("--bsz", type=int, default=100,
                        help="Batch size.")
    parser.add_argument("--fcLayers", nargs="?", default="[1024, 512, 256, 128, 64]",
                        help="Size of each layer. Note that the first layer is the "
                             "concatenation of user and item embeddings. So fcLayers[0]/2 is the embedding size.")
    parser.add_argument("--nNeg", type=int, default=4,
                        help="Number of negative instances to pair with a positive instance.")
    parser.add_argument("--lr", type=float, default=0.00001,
                        help="Learning rate.")
    parser.add_argument("--optim", nargs="?", default="adam",
                        help="Specify an optimizer: adagrad, adam, rmsprop, sgd")
    parser.add_argument("--enable_lat", nargs="?", default=True)
    parser.add_argument("--epsilon", nargs="?", default=0.5)
    parser.add_argument("--alpha", nargs="?", default=1)
    parser.add_argument("--pro_num", nargs="?", default=25)
    parser.add_argument("--decay_factor", nargs="?", default=1.0)
    parser.add_argument("--layerlist", nargs="?", default="all")
    parser.add_argument("--adv_type", nargs="?", default="mim", choices=['fgsm', 'bim', 'pgd',"mim"])
    parser.add_argument("--norm", nargs="?", default="linf", choices=['linf', 'l2'])
    return parser.parse_args()

def cal_lp_norm(tensor,p,dim_count):
    tmp = tensor
    for i in range(1,dim_count):
        tmp = torch.norm(tmp,p=p,dim=i,keepdim=True) #torch.Size([100, 1])
    
    
    return torch.clamp_min(tmp, 1e-8)



import argparse
import os
import time

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class MLP(nn.Module):
    def __init__(self, fc_layers, user_matrix, item_matrix,device):
        super(MLP, self).__init__()

        # Define layers
        self.device = device
        self.user_fc = nn.Linear(item_matrix.size(0), fc_layers[0]//2)
        self.item_fc = nn.Linear(user_matrix.size(0), fc_layers[0]//2)
        layers = []
        for l1, l2 in zip(fc_layers[:-1], fc_layers[1:]):
            layers += [nn.Linear(l1, l2), nn.ReLU(inplace=True)]
        self.fcs = nn.Sequential(*layers)
        self.final = nn.Sequential(nn.Linear(fc_layers[-1], 1), nn.Sigmoid())

        # Register user and item matrices as buffers
        self.register_buffer("user_matrix", user_matrix)
        self.register_buffer("item_matrix", item_matrix)
        print(self)

    def forward(self, user, item):
        userInput = self.user_matrix[user, :].to(self.device)          # (B, 3706)
        itemInput = self.item_matrix[item, :].to(self.device)          # (B, 6040)
        user_vector = self.user_fc(userInput).to(self.device)           # (B, fcLayers[0]//2)
        item_vector = self.item_fc(itemInput).to(self.device)           # (B, fcLayers[0]//2)


        # Concatenate user and item vectors
        x = torch.cat((user_vector, item_vector), -1)

        # Pass through fully connected layers
        x = self.fcs(x)

        # Final layer
        y = self.final(x)

        return y.squeeze()

    

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
    parser.add_argument("--fcLayers", nargs="?", default="[1024, 512, 256, 128, 64, 32, 16]",
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




if __name__ == "__main__":
    args = parse_args()
    fcLayers = eval(args.fcLayers)
    topK = 10
    topK1 = 20
    topK2 = 50
    topK3 = 100
    
    print("MLP arguments: %s " %(args))
    os.makedirs("pretrained", exist_ok=True)
    modelPath = f"pretrained/{args.dataset}_MLP_{time.time()}.pth"

    isCuda = torch.cuda.is_available()
    print(f"Use CUDA? {isCuda}")

    # Loading data
    t1 = time.time()
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives

    nUsers, nItems = train.shape
    print(f"Load data: #user={nUsers}, #item={nItems}, #train={train.nnz}, #test={len(testRatings)} [{time.time()-t1:.1f}s]")
    
    # Build model
    userMatrix = torch.Tensor(get_train_matrix(train))
    itemMatrix = torch.transpose(torch.Tensor(get_train_matrix(train)), 0, 1)    # 交换张良的维度，转置
   
    userMatrix, itemMatrix = userMatrix.to(device), itemMatrix.to(device)
    
    model = MLP(fcLayers, userMatrix, itemMatrix, alpha=args.alpha, epsilon=args.epsilon, pro_num=args.pro_num,\
         enable_lat=args.enable_lat, layerlist=args.layerlist, norm=args.norm, adv_type=args.adv_type, decay_factor=args.decay_factor, device=device)
    
    model.to(device)
    torch.save(model.state_dict(), modelPath)
    
    optimizer = get_optimizer(args.optim, args.lr, model.parameters())
    criterion = torch.nn.BCELoss()

    # Check Init performance
    t1 = time.time()
    hits, ndcgs,_,_,_,_,_,_,_,_,_,_ = evaluate_model(model, testRatings, testNegatives, topK,topK1,topK2,topK3, num_thread=1, device=device)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print(f"Init: HR={hr:.4f}, NDCG={ndcg:.4f} [{time.time()-t1:.1f}s]")
    bestHr, bestNdcg, bestEpoch = hr, ndcg, -1
    
    # Train model
    model.train()
    if args.enable_lat:
        model.zero_reg()

    for epoch in range(args.epochs):
        t1 = time.time()
        # Generate training instances
        userInput, itemInput, labels = get_train_instances(train, args.nNeg)
        dst = BatchDataset(userInput, itemInput, labels)
        ldr = torch.utils.data.DataLoader(dst, batch_size=args.bsz, shuffle=True, drop_last=True)
        losses = AverageMeter("Loss")
        
        
        for ui, ii, lbl in ldr:

            ui, ii, lbl = ui.to(device), ii.to(device), lbl.to(device)
            
            for iter in range(args.pro_num):   
                ri = model(ui, ii).squeeze()
                # if not (ri.data.cpu().numpy().all()>=0. and ri.data.cpu().numpy().all()<=1.):
                # print(ri.data)
                # if not (lbl.data.cpu().numpy().all()>=0. and lbl.data.cpu().numpy().all()<=1.):
                # print(lbl.data)
                loss = criterion(ri, lbl)
                # Update model and loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                model.save_grad() 
                losses.update(loss.item(), lbl.size(0))

        print(f"Epoch {epoch+1}: Loss={losses.avg:.4f} [{time.time()-t1:.1f}s]")

        # Evaluation
        t1 = time.time()
        hits, ndcgs,_,_,_,_,_,_,_,_,_,_ = evaluate_model(model, testRatings, testNegatives, topK,topK1,topK2,topK3, num_thread=1, device=device)
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        print(f"Epoch {epoch+1}: HR={hr:.4f}, NDCG={ndcg:.4f} [{time.time()-t1:.1f}s]")
        if hr > bestHr:
            bestHr, bestNdcg, bestEpoch = hr, ndcg, epoch
            torch.save(model.state_dict(), modelPath)

    print(f"Best epoch {bestEpoch+1}:  HR={bestHr:.4f}, NDCG={bestNdcg:.4f}")
    print(f"The best DMF model is saved to {modelPath}")

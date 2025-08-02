import random
import torch
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
EPS = 1E-20

loss_function = nn.BCEWithLogitsLoss()
# random.seed(123)
# random_float = random.uniform(0, 1)
# print(random_float)

def diff_in_weights(model, proxy):
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
        if len(old_w.size()) <= 1:
            continue
        if 'weight' in old_k:
            diff_w = new_w - old_w
            diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
    return diff_dict


def add_into_weights(model, diff, seeds,coeff=1.0,ratio=0.5): # 使用pgd更新模型参数
    names_in_diff = diff.keys()
    i = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            # T = random.uniform(0.8, 1.2)
            # random_float = random.uniform(0, 1)
            # if name in names_in_diff and random_float < ratio:
            #     param.add_(coeff * diff[name])
            if name in names_in_diff :
                if seeds[i] == 1:
                    param.add_(coeff * diff[name])
                i = i+1


class AdvWeightPerturb(object):
    def __init__(self, model, proxy, proxy_optim, gamma):
        super(AdvWeightPerturb, self).__init__()
        self.model = model
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.gamma = gamma

    def calc_awp(self, inputs_adv, targets):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()
        
        loss = - F.cross_entropy(self.proxy(inputs_adv), targets)

        self.proxy_optim.zero_grad()
        loss.backward()
        self.proxy_optim.step()

        # the adversary weight perturb
        diff = diff_in_weights(self.model, self.proxy)
        return diff

    def perturb(self, diff):
        add_into_weights(self.model, diff, coeff=1.0 * self.gamma)

    def restore(self, diff):
        add_into_weights(self.model, diff, coeff=-1.0 * self.gamma)


class AdvWeightPerturb1(object):
    def __init__(self, model, proxy, proxy_optim, gamma):
        super(AdvWeightPerturb1, self).__init__()
        self.model = model
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.gamma = gamma

    def calc_awp(self, inputs_adv_u, inputs_adv_i,inputs_adv_lab,attack_method,pro_num):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()
        criterion = torch.nn.BCELoss()
        # loss = - F.cross_entropy(self.proxy(inputs_adv_u, inputs_adv_i), inputs_adv_lab)
        # output,_ = self.proxy(inputs_adv_u, inputs_adv_i)
        if attack_method == 'fgsm':
            output = self.proxy(inputs_adv_u, inputs_adv_i)
            # yc = output.squeeze()
            loss = -loss_function(output, inputs_adv_lab)
            self.proxy_optim.zero_grad()
            loss.backward()
            self.proxy_optim.step()
        elif attack_method == 'pgd':
            for i in range(pro_num):
                _,_,_,_,_,_,_,output = self.proxy(inputs_adv_u, inputs_adv_i,inputs_adv_lab)
                yc = output.squeeze()
                loss = -criterion(yc, inputs_adv_lab)
                self.proxy_optim.zero_grad()
                loss.backward()
                self.proxy_optim.step()
        diff = diff_in_weights(self.model, self.proxy)
        return diff


    def perturb(self, diff,seeds):
        add_into_weights(self.model, diff, seeds,coeff=1.0 * self.gamma)

    def restore(self, diff,seeds):
        add_into_weights(self.model, diff, seeds, coeff=-1.0 * self.gamma)



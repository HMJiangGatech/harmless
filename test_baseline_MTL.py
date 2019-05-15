#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 13:07:56 2019

@author: yujia
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 14:24:02 2019

@author: yujia
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 12:13:13 2019

@author: yujia
"""

import numpy as np
import math
from scipy.special import digamma
import networkx as nx
from sklearn.metrics import roc_curve, auc

import os

import torch
import torch.nn

from utils import *
from Hawkes_model import *

seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
    
Tensor = torch.FloatTensor

    
def evaluate(x, pred, mu, alpha, w):
    delta = pred-x
    delta = torch.exp(-w*delta)
    delta = alpha * w * torch.sum(delta)
    
    pos = torch.log(mu + delta)
    neg = alpha * (1- torch.exp(-w*(pred-x[:,-1])))
    NLL = mu * (pred-x[:,-1]) - pos + neg
    
    return NLL

def get_prob(x, delta_T, update_mu, update_alpha, update_w):
    delta = x[:,-1] - x
    delta = torch.exp(-update_w*delta)
    delta = update_alpha * (torch.exp(-update_w*delta_T)-1) * torch.sum(delta)
    
    prob = 1 - torch.exp(-update_mu*delta_T + delta)
    return prob
#%%
def mle(x, mu, alpha, w):
    T = 1
    reg = 1e-2
    delta = x.unsqueeze(1)-x.unsqueeze(2)
    w_t_delta = -w*delta
    w_t_delta =  w_t_delta.clone().masked_fill_(delta<=0, -float('inf'))
    delta1 = torch.exp(w_t_delta)
    delta3 = alpha * w * torch.sum(delta1, dim=1)

    pos = torch.log(mu + delta3)
    neg = alpha * (1- torch.exp(-w*(T-x)))
    NLL = mu * T - torch.sum(pos) +torch.sum(neg)

    loss = NLL+ reg* (-torch.log(mu)-torch.log(alpha)-torch.log(w))
    return loss
    
def initialize(data, T, lr, device):   
    
    

    tweets = [Tensor(item).unsqueeze(0) for item in data['tweets']]
    val_tweets = data['val_tweets']
    N = len(tweets)

    theta = np.exp(np.random.normal(size=(1, 3)))*0.1
    models_param = torch.randn((N, 3), device=device, requires_grad=True)
    models_param.data = models_param.data*0.01
        
    common_model_param = torch.from_numpy(theta).float()
    common_model_param.requires_grad=True
    
    params = [models_param, common_model_param]
    optimizer = torch.optim.SGD(params, lr=lr)
            
    test_level = 0.5
    num_test = int(N*test_level)
    num_val = N - num_test
    np.random.seed(101)
    indexes = np.random.permutation(N)
    index_val = indexes[:num_val]
    index_test = indexes[num_val:]
  
    return models_param, common_model_param, optimizer, tweets, val_tweets, index_val, index_test


device = 'cpu'
num_iter = 1000
reg_mtl = 0.1
lr = 1e-2 # might need to change w.r.t. N

print('Loading data...')

#G, tweets, val_tweets = load_data(PATH)
#G_matrix = nx.to_numpy_matrix(G)
#G_matrix = np.asarray(G_matrix)

PATH = '../data_preprocess/PoPPy/data/'  
G, tweets, val_tweets = load_linkedin_data(PATH)
G_matrix = nx.to_numpy_matrix(G)
G_matrix = np.asarray(G_matrix) 
data = {'matrix': G_matrix, 'G': G, 'tweets': tweets,'val_tweets': val_tweets  }

print('Initializing...')
models_param, common_model_param, optimizer, tweets, val_tweets, index_val, index_test = initialize(data, 1, lr, device)
N = len(tweets)


#parameter_list = []
loss_list = []
eval_list = []
#%%
for it in range(num_iter):
    
    print('Performing update', it)
    optimizer.zero_grad()
    hat_model_param = models_param + common_model_param
    accum_loss = 0
    for i, tweet in enumerate(tweets):
        
        loss = mle(tweet, hat_model_param[i,0], hat_model_param[i,1], hat_model_param[i,2])
        accum_loss += loss
    accum_loss /= N 
    accum_loss += reg_mtl* models_param.norm()
    accum_loss.backward()
    optimizer.step()
    
#    if error_flag:
#        print('Encountered invalid value, stopping the iterations')
#        break
    loss_list.append(accum_loss.data.item())
    
    print('Iteration:', it, 'Loss:', loss.data.item())

    if it % 1 == 0:
        with torch.no_grad():
            accum_nll1 = 0
            hat_model_param = models_param + common_model_param
            for i in index_val:
                nll = evaluate(tweets[i], val_tweets[i], hat_model_param[i,0], hat_model_param[i,1], hat_model_param[i,2])
                accum_nll1 += nll.data.item()
            accum_nll2 = 0
            for i in index_test:
                nll = evaluate(tweets[i], val_tweets[i], hat_model_param[i,0], hat_model_param[i,1], hat_model_param[i,2])
                accum_nll2 += nll.data.item()
                
                
                
        nll = (accum_nll1 + accum_nll2)/N
        eval_list.append(nll)
        
        print('Iteration:', it, 'Eval NLL:', nll)
#        hawkes_models.show_param()
        
#        hawkes_models.train()

#%%
# note there are global variable used inside
def get_roc_auc(delta_T):
    prob_list = []
    truth_list = []
    
    for i in index_test:
        if bool(val_tweets[i]-tweets[i][0][-1]<=delta_T):
            truth_list.append(1)
        else:
            truth_list.append(0)
        with torch.no_grad():
            prob = get_prob(tweets[i], delta_T, hat_model_param[i,0], hat_model_param[i,1], hat_model_param[i,2])
        prob_list.append(prob.data.item())
        
    fpr, tpr, _ = roc_curve(truth_list, prob_list)
    
    roc_auc = auc(fpr, tpr)
    
    return roc_auc, fpr, tpr
        
delta_T_list = list(np.arange(1,11,1))
delta_T_list = list(np.arange(0.25,1,0.25))+delta_T_list
delta_T_list = [item/47.7753 for item in delta_T_list]


auc_list = []
fpr_list = []
tpr_list = []
for delta_T in delta_T_list:
    roc_auc, fpr, tpr = get_roc_auc(delta_T)
    auc_list.append(roc_auc)
    fpr_list.append(fpr)
    tpr_list.append(tpr)

result_PATH = os.path.join('../result', 'baselineMTL')
if not os.path.isdir(result_PATH):
    os.makedirs(result_PATH)

file_name = 'linkedin_seed'+str(seed)+'_lr'+str(1e-2) \
            +'_iter'+str(num_iter) \
            + '_method_baseline2_init_random.txt'
with open(os.path.join(result_PATH, file_name), 'w') as f:
    for loss in loss_list:
        f.write(str(loss)+', ')
    for nll in eval_list:
        f.write(str(nll)+', ')
    f.write('\n')
    f.write('delta_T\n')
    for delta_T in delta_T_list:
        f.write(str(delta_T)+', ')
    f.write('\n')
    f.write('auc\n')
    for roc_auc in auc_list:
        f.write(str(roc_auc)+', ')
        
#%%
import matplotlib.pyplot as plt
plt.plot(eval_list)
plt.show()
#%%
for delta_T, fpr, tpr in zip(delta_T_list, fpr_list, tpr_list):
    plt.plot(fpr, tpr, label='$\Delta T$='+str(delta_T))
plt.legend(fontsize=15)
plt.show()
#%%
plt.plot(delta_T_list, auc_list)
plt.ylim([0,1])
plt.show()
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

import os

import torch
import torch.nn

from utils import *
from Hawkes_model import *

seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
    
    
def initialize(data, alpha0, K, T,method, device):
    
    hawkes_models = Hawkes_models(data,T,K,method,lr=1e-3, inner_lr=1e-3, device=device, init='random')

  
    return hawkes_models

def update_parameter(hawkes_models, N, K):
    

    L = hawkes_models.compute_loss()
    
    if np.isnan(np.sum(L)):
        return 1, None
#    print('L: ', L)
    weights = np.ones((N, K))/(N*K)
    loss = hawkes_models.update_theta(weights)
        
    return 0, loss


#PATH = '../twitter/'
device = 'cpu'
num_iter = 500
K = 1
alpha0 = np.ones(K)/K

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
hawkes_models = initialize(data, alpha0, K, 1, 'mle', device)
N = len(tweets)


#parameter_list = []
loss_list = []
eval_list = []
weights = np.ones((N, K))
#%%
for it in range(num_iter):
    
    print('Performing update', it)
    
    error_flag, loss = update_parameter(hawkes_models, N, K)
    
    if error_flag:
        print('Encountered invalid value, stopping the iterations')
        break
    
#    parameter_list.append(parameter.copy())
    loss_list.append(loss)
    
    print('Iteration:', it, 'Loss:', loss)

    if it % 1 == 0:
#        hawkes_models.eval()
        nll = hawkes_models.evaluate(weights)
        eval_list.append(nll)
        
        print('Iteration:', it, 'Eval NLL:', nll)
#        hawkes_models.show_param()
        
#        hawkes_models.train()

#%%
        
delta_T_list = np.arange(0.025,0.2,0.025)
auc_list = []
fpr_list = []
tpr_list = []
for delta_T in delta_T_list:
    auc, fpr, tpr = hawkes_models.get_roc_auc(np.ones(shape=(N,1)), delta_T)
    auc_list.append(auc)
    fpr_list.append(fpr)
    tpr_list.append(tpr)

result_PATH = os.path.join('../result', 'baseline2')
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
    for auc in auc_list:
        f.write(str(auc)+', ')
        
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
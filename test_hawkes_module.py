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
    
    hawkes_models = Hawkes_models(data,T,K,method,lr=2e-3, inner_lr=2e-3, device=device)

  
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
weights = np.ones((N, K))/(N*K)
#%%
for it in range(num_iter):
    
    print('Performing update', it)
    
    error_flag, loss = update_parameter(hawkes_models, N, K)
    
    if error_flag:
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
import matplotlib.pyplot as plt
plt.plot(eval_list)
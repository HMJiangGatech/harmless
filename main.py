#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 12:13:13 2019

@author: yujia
"""

'''
TO DO:
    1. check clusters
    2. check reptile, fomaml, and maml


'''

import numpy as np
import math
from scipy.special import digamma
import networkx as nx

import os

import torch
import torch.nn

from utils import *
from Hawkes_model import *
from update_param import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='linkedin')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--inner_lr', type=float, default=1e-2)
parser.add_argument('--K', type=int, default=2)
parser.add_argument('--max_iter', type=int, default=1000)
parser.add_argument('--pretrain_iter', type=int, default=0)
parser.add_argument('--method', type=str, default='maml', help='mle | maml | fomaml | reptile')
parser.add_argument('--init_theta', type=str, default='random', help='uniform | random ')

opt = parser.parse_args()

seed = opt.seed
np.random.seed(seed)
torch.manual_seed(seed)
    
    
def initialize(data, alpha0, K, T, method, device, lr, inner_lr, init_theta):
    G_matrix, tweets = data['matrix'], data['tweets']
    
    parameter = {'alpha': None, 
                 'gamma': None,
                 'phi': None,
                 'psi': None,
                 'Hawkes_models': [],
                 'B': None,
                 'alpha0': alpha0,
                 'N': len(tweets),
                 'K': K}
    
    N = len(tweets)
#    parameter['alpha'] = alpha0.copy()
    parameter['gamma'] = np.random.dirichlet(parameter['alpha0'], size=(N))
#    parameter['gamma'] = np.exp(np.random.normal(size=(N,K)))
#    parameter['gamma'] = parameter['gamma']/np.sum(parameter['gamma'], axis=-1, keepdims=True)
    
    parameter['phi'] = np.random.dirichlet(parameter['alpha0'], size=(N, N))
    
    parameter['psi'] = np.random.dirichlet(parameter['alpha0'], size=(N, N))
    
    hawkes_models = Hawkes_models(data,T,K,method, lr=lr, inner_lr=inner_lr, device=device, init = init_theta)
    
    phi_psi = np.expand_dims(parameter['phi'], axis=-1) * np.expand_dims(parameter['psi'], axis=-2)
    num_B = np.sum(phi_psi*np.expand_dims(np.expand_dims(G_matrix, axis=-1), axis=-1) , axis=(0,1))
    
#    rho_num = np.sum((1-G_matrix)*np.sum(phi_psi, axis=(2,3)))
#    rho_don = np.sum(phi_psi)
    don_B = np.sum(phi_psi, axis=(0,1))
    parameter['B'] = (num_B/don_B).clip(1e-10, 1-1e-10)
    
    parameter['rho'] = (1.-np.sum(G_matrix)/(N*N))
  
    return parameter, hawkes_models


if __name__ == '__main__':
    
#    PATH = '../twitter/'
    device = 'cpu'
    pretrain_iter = opt.pretrain_iter
    num_iter = opt.max_iter
    K = opt.K
    alpha0 = np.ones(K)
    method = opt.method
    
    print('Loading data...')
    
    if opt.data == 'twitter':
        PATH = 'data/twitter'
        G, tweets, val_tweets = load_data(PATH)
        G_matrix = nx.to_numpy_matrix(G)
        G_matrix = np.asarray(G_matrix) + np.eye(len(G_matrix))
    elif opt.data == 'linkedin':
        PATH = 'data/linkedin'  
        G, tweets, val_tweets = load_linkedin_data(PATH)
        G_matrix = nx.to_numpy_matrix(G)
        G_matrix = np.asarray(G_matrix)
    elif opt.data == 'mathoverflow':
        PATH = 'data/mathoverflow'
        G, tweets, val_tweets = load_overflow_data(PATH)
        G_matrix = nx.to_numpy_matrix(G)
        G_matrix = np.asarray(G_matrix) + np.eye(len(G_matrix))
    elif opt.data == 'stackoverflow':
        PATH = 'data/stackoverflow'
        G, tweets, val_tweets = load_overflow_data(PATH, False)
        G_matrix = nx.to_numpy_matrix(G)
        G_matrix = np.asarray(G_matrix) + np.eye(len(G_matrix))

    data = {'matrix': G_matrix, 'G': G, 'tweets': tweets, 'val_tweets': val_tweets }
    
    print('Initializing...')
    parameter, hawkes_models = initialize(data, alpha0, K, 1, method, device, opt.lr, opt.inner_lr, opt.init_theta)
    
    parameter = pretrain(data, parameter, pretrain_iter, opt.lr)
    
    
#    parameter_list = []
    loss_list = []
    eval_list = []
    #%%
    for it in range(num_iter):
        
        print('Performing update', it)
        
        error_flag, loss = update_parameter(data, parameter, hawkes_models, opt.lr)
        
        if error_flag:
            break
        
#        parameter_list.append(parameter.copy())
        loss_list.append(loss)
        
        print('Iteration:', it, 'Loss:', loss)
    
        if it % 1 == 0:
    #        hawkes_models.eval()
            nll = hawkes_models.evaluate(parameter['gamma'])
            eval_list.append(nll)
            
            print('Iteration:', it, 'Eval NLL:', nll)
            
    #        hawkes_models.train()
    
        
    #%%
        
    import matplotlib.pyplot as plt
    plt.plot(eval_list)
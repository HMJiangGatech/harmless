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
import pickle

import json

import os

import torch
import torch.nn

from utils import *
from Hawkes_model import *
from update_param import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='stackoverflow')
parser.add_argument('--result_path', type=str, default='../result')
parser.add_argument('--read_checkpoint', type=str, default=None)#'../result/our_checkpoint/linkedin_seed1_lr0.01_innerlr0.01_K2_pretrain0_iter10_method_maml_init_random.txt')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--lr', type=float, default=4e-4)
parser.add_argument('--inner_lr', type=float, default=4e-4)
parser.add_argument('--K', type=int, default=2)
parser.add_argument('--max_iter', type=int, default=1000)
parser.add_argument('--pretrain_iter', type=int, default=0)
parser.add_argument('--method', type=str, default='mle', help='mle | maml | fomaml | reptile')
parser.add_argument('--init_theta', type=str, default='random', help='uniform | random ')



opt = parser.parse_args()
#%%
seed = opt.seed
np.random.seed(seed)
torch.manual_seed(seed)


def initialize(data, alpha0, K, T, method, device, lr, inner_lr, init_theta, read_checkpoint):
    G_matrix, tweets = data['matrix'], data['tweets']
    if read_checkpoint:
        with open(read_checkpoint, 'rb') as f:
            parameter = f.read()

        parameter = pickle.loads(parameter)
        hawkes_models = parameter['Hawkes_models']

    else:

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
        parameter['gamma'] = np.random.dirichlet(parameter['alpha0'], size=(N))
        parameter['phi'] = np.random.dirichlet(parameter['alpha0'], size=(N, N))
        parameter['psi'] = np.random.dirichlet(parameter['alpha0'], size=(N, N))

        hawkes_models = Hawkes_models(data,T,K,method, lr=lr, inner_lr=inner_lr, device=device, init = init_theta)

        phi_psi = np.expand_dims(parameter['phi'], axis=-1) * np.expand_dims(parameter['psi'], axis=-2)
        num_B = np.sum(phi_psi*np.expand_dims(np.expand_dims(G_matrix, axis=-1), axis=-1) , axis=(0,1))

        don_B = np.sum(phi_psi, axis=(0,1))
        parameter['B'] = (num_B/don_B).clip(1e-10, 1-1e-10)

        parameter['rho'] = (1.-np.sum(G_matrix)/(N*N))

    return parameter, hawkes_models


main():

    # GPU is slower, bacause the major optimization part does not involve large matrix computation
    device = 'cpu' 
    pretrain_iter = opt.pretrain_iter
    num_iter = opt.max_iter
    K = opt.K
    alpha0 = np.ones(K)
    method = opt.method

    print('Loading data...')

    if opt.data == 'linkedin':
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
    elif opt.data == '911':
        PATH = 'data/911'
        G, tweets, val_tweets = load_911_data(PATH)
        G_matrix = nx.to_numpy_matrix(G)
        G_matrix = np.asarray(G_matrix) 

    data = {'matrix': G_matrix, 'G': G, 'tweets': tweets, 'val_tweets': val_tweets }

    print('Initializing...')
    parameter, hawkes_models = initialize(data, alpha0, K, 1, method, device, opt.lr, opt.inner_lr, opt.init_theta, opt.read_checkpoint)

    parameter = pretrain(data, parameter, pretrain_iter, opt.lr)

    result_PATH = os.path.join(opt.result_path, 'our_result')
    if not os.path.isdir(result_PATH):
        os.makedirs(result_PATH)

    temp_file_name = 'temp_' + str(opt.data)+'_seed'+str(opt.seed)+'_lr'+str(opt.lr)+'_innerlr'+str(opt.inner_lr) \
                +'_K'+str(opt.K)+'_pretrain'+str(opt.pretrain_iter)+'_iter'+str(opt.max_iter) \
                + '_method_'+opt.method+'_init_'+opt.init_theta+'.txt'

    loss_list = []
    eval_list = []
    eval_list_val = []
    eval_list_test = []

    with open(os.path.join(result_PATH, temp_file_name), 'w') as f:
        for it in range(num_iter):

            print('Performing update', it)

            error_flag, loss = update_parameter(data, parameter, hawkes_models, opt.lr)

            if error_flag:
                break

            loss_list.append(loss)

            print('Iteration:', it, 'Loss:', loss)

            if it % 1 == 0:
                hawkes_models.eval()
                nll_val, nll_test, nll_all = hawkes_models.evaluate(parameter['gamma'])
                eval_list.append(nll_all)
                eval_list_val.append(nll_val)
                eval_list_test.append(nll_test)

                print('Iteration:', it, 'Eval NLL:', nll_val)


                f.write(str(loss)+', ')
                f.write(str(nll_all)+', ')
                f.write(str(nll_val)+', ')
                f.write(str(nll_test)+', ')
                f.write('\n')

                hawkes_models.train()




    checkpoint_PATH = os.path.join(opt.result_path, 'our_checkpoint')
    if not os.path.isdir(checkpoint_PATH):
        os.makedirs(checkpoint_PATH)

    with open(os.path.join(checkpoint_PATH, file_name), 'wb') as f:
        parameter['Hawkes_models'] = hawkes_models
        f.write(pickle.dumps(parameter))

if __name__ == '__main__':
    main()




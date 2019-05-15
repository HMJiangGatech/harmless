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
import random
import json
import math
import copy
import pickle
from scipy.special import digamma
import networkx as nx

import os

import torch
import torch.nn

from utils import *
from Hawkes_model import *
from update_param import *

import generator as dataset

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='mmb_hard', \
                    help='balanced_tree | balanced_treev2 | barbell | mmb | mmb_hard')
parser.add_argument('--result_path', type=str, default=None)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--inner_lr', type=float, default=1e-5)
parser.add_argument('--K', type=int, default=2)
parser.add_argument('--max_iter', type=int, default=1000)
parser.add_argument('--pretrain_iter', type=int, default=0)
parser.add_argument('--method', type=str, default='maml', help='mle | maml | fomaml | reptile')
parser.add_argument('--init_theta', type=str, default='uniform', help='uniform | random ')
parser.add_argument('--hardgamma', action='store_true', help='use hard gamma in VI')
## mmb
parser.add_argument('--mmb_nodes', type=int, default=50)
parser.add_argument('--mmb_clusters', type=int, default=6)
parser.add_argument('--bjk', type=float, default=1)
parser.add_argument('--bkk', type=float, default=5)

parser.add_argument('--verbose', action='store_true', help='verbose')

opt = parser.parse_args()
#%%
seed = opt.seed
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)


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
    parameter['alpha'] = np.random.dirichlet(parameter['alpha0'], size=(N))#alpha0.copy()
    # parameter['alpha'] = np.ones([N,K])*2*N/K
    parameter['gamma'] = np.random.dirichlet(parameter['alpha0'], size=(N))
    # parameter['gamma'] = np.ones([N,K])/K
    if opt.hardgamma:
        parameter['gamma'] = (parameter['gamma']>=np.expand_dims(parameter['gamma'].max(1),axis=1)).astype(float)
    # parameter['gamma'] = np.exp(np.random.normal(size=(N,K)))
    # parameter['gamma'] = parameter['gamma']/np.sum(parameter['gamma'], axis=-1, keepdims=True)

    parameter['phi'] = np.random.dirichlet(parameter['alpha0'], size=(N, N))

    parameter['psi'] = np.random.dirichlet(parameter['alpha0'], size=(N, N))

    hawkes_models = Hawkes_models(data,T,K,method, lr=lr, inner_lr=inner_lr, device=device, init = init_theta)

    phi_psi = np.expand_dims(parameter['phi'], axis=-1) * np.expand_dims(parameter['psi'], axis=-2)
    num_B = np.sum(phi_psi*np.expand_dims(np.expand_dims(G_matrix, axis=-1), axis=-1) , axis=(0,1))

    # rho_num = np.sum((1-G_matrix)*np.sum(phi_psi, axis=(2,3)))
    # rho_don = np.sum(phi_psi)
    don_B = np.sum(phi_psi, axis=(0,1))
    parameter['B'] = (num_B/don_B).clip(1e-10, 1-1e-10)

    parameter['rho'] = (1.-np.sum(G_matrix)/(N*N))

    return parameter, hawkes_models


if __name__ == '__main__':

    result_PATH = os.path.join('result/', opt.data+"_Kt"+str(opt.mmb_clusters)+\
                            "_bjk"+str(opt.bjk)+"_bkk"+str(opt.bkk)+"_"+str(opt.seed))
    if not os.path.isdir(result_PATH):
        os.makedirs(result_PATH)

    device = 'cpu'
    pretrain_iter = opt.pretrain_iter
    num_iter = opt.max_iter
    K = opt.K
    alpha0 = np.ones(K)
    method = opt.method

    print('Loading data...')

    data_path = os.path.join(result_PATH,'data.pkl')
    if os.path.exists(data_path):
        with open(data_path, 'rb') as datafile:
            data_dictionary=pickle.load(datafile)
        G = data_dictionary['G']
        tweets = data_dictionary['tweets']
        val_tweets = data_dictionary['val_tweets']
        true_param = data_dictionary['true_param']
        true_member = data_dictionary['true_member']
        G_pos = data_dictionary['G_pos']
    else:
        if opt.data == 'balanced_tree':
            G, tweets, val_tweets, true_param, true_member, G_pos = \
                    dataset.balanced_tree(r=3,h=3,h_thr=[2],path=result_PATH)
        if opt.data == 'balanced_treev2':
            G, tweets, val_tweets, true_param, true_member, G_pos = \
                    dataset.balanced_treev2(tr=3,r=2,h=3,path=result_PATH)
        if opt.data == 'barbell':
            G, tweets, val_tweets, true_param, true_member, G_pos = \
                    dataset.barbell(tr=3,m1=5,m2=2,path=result_PATH)
        if opt.data == 'mmb':
            G, tweets, val_tweets, true_param, true_member, G_pos = \
                    dataset.mmb(nodes=opt.mmb_nodes,clusters=opt.mmb_clusters,bjk=opt.bjk,bkk=opt.bkk,hardedge=False,path=result_PATH)
        if opt.data == 'mmb_hard':
            G, tweets, val_tweets, true_param, true_member, G_pos = \
                    dataset.mmb(nodes=opt.mmb_nodes,clusters=opt.mmb_clusters,bjk=opt.bjk,bkk=opt.bkk,hardedge=True,path=result_PATH)
        with open(data_path, 'wb') as datafile:
            pickle.dump({'G': G, 'tweets': tweets, 'val_tweets':val_tweets, \
                        'true_param':true_param, 'true_member':true_member, 'G_pos':G_pos}, datafile)
    G_matrix = nx.to_numpy_matrix(G)
    G_matrix = np.asarray(G_matrix) + np.eye(len(G_matrix))

    T_max = max(val_tweets)*1.05
    for i in tweets:
        for j in range(len(i)):
            i[j] /= T_max
    for i in range(len(val_tweets)):
        val_tweets[i] /= T_max

    result_PATH = os.path.join(result_PATH,(opt.result_path+"_" if opt.result_path is not None else "")+'lr'+str(opt.lr)+'_innerlr'+str(opt.inner_lr) \
                +'_K'+str(opt.K)+'_pretrain'+str(opt.pretrain_iter)+'_iter'+str(opt.max_iter) \
                + '_method_'+opt.method+'_init_'+opt.init_theta+'_hardgamma_'+str(opt.hardgamma))
    if not os.path.isdir(result_PATH):
        os.makedirs(result_PATH)

    with open(os.path.join(result_PATH, "args.json"), 'w') as f:
        json.dump(opt.__dict__, f, indent=2)

    # Loading script ( for future use maybe)
    # parser = ArgumentParser()
    # args = parser.parse_args()
    # with open('commandline_args.txt', 'r') as f:
    #     args.__dict__ = json.load(f)

    if not os.path.isdir(result_PATH):
        os.makedirs(result_PATH)

    data = {'matrix': G_matrix, 'G': G, 'tweets': tweets, 'val_tweets': val_tweets }

    print('Initializing...')
    parameter, hawkes_models = initialize(data, alpha0, K, 1, method, device, opt.lr, opt.inner_lr, opt.init_theta)

    parameter = pretrain(data, parameter, pretrain_iter, opt.lr)

    # parameter_list = []
    loss_list = []
    eval_list = []
    #%%
    for it in range(num_iter):

        print('Performing update', it)
        hawkes_models.show_param()

        error_flag, loss = update_parameter(data, parameter, hawkes_models, opt.lr, hardgamma=opt.hardgamma, verbose=opt.verbose)

        if error_flag:
            break

        # parameter_list.append(parameter.copy())
        loss_list.append(loss)

        print('Iteration:', it, 'Loss:', loss)

        if it % 1 == 0:
            hawkes_models.eval()
            nll = hawkes_models.evaluate(parameter['gamma'])
            eval_list.append(nll)

            print('Iteration:', it, 'Eval NLL:', nll)

            hawkes_models.train()


    #%%

    delta_T_list = np.arange(0.025,0.2,0.025)
    auc_list = []
    fpr_list = []
    tpr_list = []
    for delta_T in delta_T_list:
        auc, fpr, tpr = hawkes_models.get_roc_auc(parameter['gamma'], delta_T)
        auc_list.append(auc)
        fpr_list.append(fpr)
        tpr_list.append(tpr)

    file_name = 'loss.txt'
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
    plt.savefig(fname=os.path.join(result_PATH, "loss.pdf"), bbox_inches='tight',format="pdf", dpi = 300)
    plt.close()

    #%%

    for delta_T, fpr, tpr in zip(delta_T_list, fpr_list, tpr_list):
        plt.plot(fpr, tpr, label='$\Delta T$='+str(delta_T))
    plt.legend(fontsize=15)
    plt.savefig(fname=os.path.join(result_PATH, "roc.pdf"), bbox_inches='tight',format="pdf", dpi = 300)
    plt.close()

    plt.plot(delta_T_list, auc_list)
    plt.ylim([0,1])
    plt.savefig(fname=os.path.join(result_PATH, "auc.pdf"), bbox_inches='tight',format="pdf", dpi = 300)
    plt.close()

    # draw membership
    if opt.verbose:
        mixed_membership = copy.copy(parameter["alpha"])
        num_cluster = len(mixed_membership[0])
        for i in mixed_membership:
            _sum = sum(i)
            for j in range(num_cluster):
                i[j] /= _sum
        mixed_membership = np.array(mixed_membership)
        if num_cluster > 10:
            print("Warning! Don't have enough colors, random generate color map")
            color_map = [(random.random(),random.random(),random.random()) for i in range(num_cluster)]
        else:
            color_map = plt.get_cmap('tab10').colors[:num_cluster]
        color_map = np.array(color_map)
        mixed_color = np.matmul(mixed_membership, color_map)

        argmax_color = color_map[mixed_membership.argmax(1),:]
        dataset.draw_net(G,argmax_color,result_PATH,filename="max_membership",pos=G_pos,html=False)
        gamma_color = color_map[np.array(parameter["gamma"]).argmax(1),:]
        dataset.draw_net(G,gamma_color,result_PATH,filename="vi_membership",pos=G_pos,html=False)
        dataset.draw_net(G,mixed_color,result_PATH,filename="mixed_membership",pos=G_pos,html=False)
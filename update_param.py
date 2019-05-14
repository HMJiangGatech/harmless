#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 19:12:26 2019

@author: yujia
"""

import numpy as np
import math
from scipy.special import digamma
import ot

import torch
import torch.nn

from sklearn.cluster import SpectralClustering

def pre_update_parameter(data, parameter, lr):
    G_matrix = data['matrix']
    N = G_matrix.shape[0]
    K = parameter['alpha'].shape[1]
    adj_matrix = G_matrix+np.eye(N)
    # embedding = sklearn.manifold.spectral_embedding(adj_matrix,n_components=10,drop_first=False)
    clustering = SpectralClustering(n_clusters=K,affinity='precomputed',assign_labels="discretize",random_state=0).fit(adj_matrix)

    parameter['alpha'] *= 0
    parameter['gamma'] *= 0
    for i,j in enumerate(clustering.labels_):
        parameter['alpha'][i,j] = 1
        parameter['gamma'][i,j] = 1
    print('alpha', np.argmax(parameter['alpha'], axis=1), np.std(np.argmax(parameter['alpha'], axis=1)))
    return 0




    # alpha_diff = np.expand_dims(parameter['alpha'],axis=1)-np.expand_dims(parameter['alpha'],axis=0)
    # print(np.einsum('ijk,ijk->ij',alpha_diff,alpha_diff).max())
    # weight = np.einsum('ij,i->ij',G_matrix,1/G_matrix.sum(1)) + \
    #         np.einsum('ij,i->ij',G_matrix-1,1/(1-G_matrix).sum(1))
    # parameter['alpha'] = parameter['alpha']*(1-lr) + lr*np.einsum('ijk,ij->ik',alpha_diff,weight)
    # print('alpha', np.argmax(parameter['alpha'], axis=1), np.std(np.argmax(parameter['alpha'], axis=1)))
    # parameter['gamma'] = np.exp(parameter['alpha'])
    # parameter['gamma'] =  parameter['gamma']/np.sum(parameter['gamma'], axis=-1, keepdims=True)
    # return 0

# def pre_update_parameter(data, parameter, lr):
#     G_matrix = data['matrix']
#     #alpha
#     parameter['alpha'] = parameter['alpha0'] \
#                         + np.sum(parameter['phi'], axis=1) \
#                         + np.sum(parameter['psi'], axis=1)
#     # print('alpha', parameter['alpha'])
#     print('alpha', np.argmax(parameter['alpha'], axis=1), np.std(np.argmax(parameter['alpha'], axis=1)))
#
#
#     digamma_alpha = digamma(parameter['alpha']) - digamma(np.sum(parameter['alpha'], axis=-1, keepdims=True))
#     exp_digamma_alpha = np.exp(digamma_alpha)
#
#
#     log_B = np.expand_dims(np.expand_dims(np.log(parameter['B']), axis=0), axis=0)
#     log_1_B = np.expand_dims(np.expand_dims(np.log(1-parameter['B']), axis=0), axis=0)
#     expand_Y = np.expand_dims(np.expand_dims(data['matrix'], axis=-1), axis=-1)
#     B_to_Y = expand_Y*log_B + (1-expand_Y)*log_1_B
#
#     for _ in range(10):
#         #phi
#         phi_right = np.sum(np.expand_dims(parameter['psi'], axis=-2)*B_to_Y, axis=-1)
#         parameter['phi'] = np.expand_dims(exp_digamma_alpha, axis=1) * np.exp(phi_right)
#         parameter['phi'] = parameter['phi']/np.sum(parameter['phi'], axis=-1, keepdims=True)
#         # print('phi', parameter['phi'][:,:,0])
#
#         #psi
#         psi_right = np.sum(np.expand_dims(parameter['phi'], axis=-1)*B_to_Y, axis=-2)
#         parameter['psi'] = np.expand_dims(exp_digamma_alpha, axis=0) * np.exp(psi_right)
#         parameter['psi'] = parameter['psi']/np.sum(parameter['psi'], axis=-1, keepdims=True)
#         # print('psi', parameter['psi'][:5,:10,:])
#
#     # alpha0
#
#     # digamma_alpha0 =  digamma(np.sum(parameter['alpha0'], axis=-1, keepdims=True)) - digamma(parameter['alpha0'])
#     # update_alpha0 = parameter['N']*digamma_alpha0 + np.sum(digamma_alpha, axis=0)
#     # parameter['alpha0'] = parameter['alpha0'] + lr*update_alpha0
#
#
#     #B
#     phi_psi = np.expand_dims(parameter['phi'], axis=-1) * np.expand_dims(parameter['psi'], axis=-2)
#     num_B = np.sum(phi_psi*np.expand_dims(np.expand_dims(G_matrix, axis=-1), axis=-1) , axis=(0,1))
#     don_B =  np.sum(phi_psi, axis=(0,1))
#
#     rho_num = np.sum((1-G_matrix)*np.sum(phi_psi, axis=(2,3)))
#     rho_don = np.sum(phi_psi)
#     don_B = (1-rho_num/rho_don) * np.sum(phi_psi, axis=(0,1))
#     parameter['B'] = (num_B/don_B)#.clip(1e-10, 1-1e-10)
#
#     # regularized B
#     # parameter['B'] = parameter['B']*0.95+0.05*np.eye(parameter['B'].shape[0])
#
#     print('B', parameter['B'])
#
#     return 0

def pretrain(data, parameter, pretrain_iter, lr):
    for it in range(pretrain_iter):
        pre_update_parameter(data, parameter, lr)

    return parameter

def update_parameter(data, parameter, hawkes_models, lr, hardgamma=False):
    G_matrix = data['matrix']
    #alpha
    parameter['alpha'] = parameter['alpha0'] \
                        + parameter['gamma'] \
                        + np.sum(parameter['phi'], axis=1) \
                        + np.sum(parameter['psi'], axis=1)
    # print('alpha', parameter['alpha'])
    print('alpha', np.std(np.argmax(parameter['alpha'], axis=1)))
    # print(parameter['alpha'][:5,:])
    # print((np.sum(parameter['phi'], axis=1) + np.sum(parameter['psi'], axis=1))[:5,:])

    #gamma
    L = hawkes_models.compute_loss()
    exp_L = np.exp(-L)

    if np.any(np.isinf(exp_L)):
        print('Overflow! Cut once ----------------------------')
        exp_L[exp_L>1e6] = 1e6

    if np.isnan(np.sum(exp_L)):
        return 1, None

    digamma_alpha = digamma(parameter['alpha']) - digamma(np.sum(parameter['alpha'], axis=-1, keepdims=True))

    exp_digamma_alpha = np.exp(digamma_alpha)
    parameter['gamma'] = exp_digamma_alpha*exp_L
    parameter['gamma'] = parameter['gamma']/np.sum(parameter['gamma'], axis=-1, keepdims=True)

    # smooth gamma

    #hardgamma
    if hardgamma:
        parameter['gamma'] = (parameter['gamma']>=np.expand_dims(parameter['gamma'].max(1),axis=1)).astype(float)

    print('gamma',  np.std(np.argmax(parameter['gamma'], axis=1)))
    #theta
    loss = hawkes_models.update_theta(parameter['gamma']/np.sum(parameter['gamma']))

    #phi
    log_B = np.expand_dims(np.expand_dims(np.log(parameter['B']), axis=0), axis=0)
    log_1_B = np.expand_dims(np.expand_dims(np.log(1-parameter['B']), axis=0), axis=0)
    expand_Y = np.expand_dims(np.expand_dims(data['matrix'], axis=-1), axis=-1)
    B_to_Y = expand_Y*log_B + (1-expand_Y)*log_1_B

    phi_right = np.sum(np.expand_dims(parameter['psi'], axis=-2)*B_to_Y, axis=-1)
    parameter['phi'] = exp_digamma_alpha * np.exp(phi_right)
    parameter['phi'] = parameter['phi']/np.sum(parameter['phi'], axis=-1, keepdims=True)

    #print('phi', parameter['phi'])

    #psi
    psi_right = np.sum(np.expand_dims(parameter['phi'], axis=-1)*B_to_Y, axis=-2)
    parameter['psi'] = exp_digamma_alpha * np.exp(psi_right)
    parameter['psi'] = parameter['psi']/np.sum(parameter['psi'], axis=-1, keepdims=True)

    #print('psi', parameter['psi'])

    # alpha0

    # digamma_alpha0 =  digamma(np.sum(parameter['alpha0'], axis=-1, keepdims=True)) - digamma(parameter['alpha0'])
    # update_alpha0 = parameter['N']*digamma_alpha0 + np.sum(digamma_alpha, axis=0)
    # parameter['alpha0'] = parameter['alpha0'] + lr*update_alpha0

    #B
    phi_psi = np.expand_dims(parameter['phi'], axis=-1) * np.expand_dims(parameter['psi'], axis=-2)
    num_B = np.sum(phi_psi*np.expand_dims(np.expand_dims(G_matrix, axis=-1), axis=-1) , axis=(0,1))
    don_B =  np.sum(phi_psi, axis=(0,1))

    # rho_num = np.sum((1-G_matrix)*np.sum(phi_psi, axis=(2,3)))
    # rho_don = np.sum(phi_psi)
    # don_B = (1-rho_num/rho_don) * np.sum(phi_psi, axis=(0,1))
    parameter['B'] = (num_B/don_B)#.clip(1e-10, 1-1e-10)

    # print('B', parameter['B'])

    return 0, loss

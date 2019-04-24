#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 18:05:12 2019

@author: yujia
"""

import os

import numpy as np
import networkx as nx

import matplotlib.pyplot as plt

from update_param import *

def initialize(data, alpha0, K, T, method, device, lr, inner_lr):
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
    
    N = len(G_matrix)
#    parameter['alpha'] = alpha0.copy()
    
    parameter['gamma'] = np.exp(np.random.normal(size=(N,K)))
    parameter['gamma'] = parameter['gamma']/np.sum(parameter['gamma'], axis=-1, keepdims=True)
    
    parameter['phi'] = np.random.dirichlet(parameter['alpha0'], size=(N, N))
#    parameter['phi'] = np.exp(np.random.normal(size=(N, N, K)))
#    parameter['phi'] = parameter['phi']/np.sum(parameter['phi'], axis=-1, keepdims=True)
    
    parameter['psi'] = np.random.dirichlet(parameter['alpha0'], size=(N, N))
#    parameter['psi'] = np.exp(np.random.normal(size=(N, N, K)))
#    parameter['psi'] = parameter['psi']/np.sum(parameter['psi'], axis=-1, keepdims=True)
    
    parameter['rho'] = (1.-np.sum(G_matrix)/(N*N))
    
    phi_psi = np.expand_dims(parameter['phi'], axis=-1) * np.expand_dims(parameter['psi'], axis=-2)
    num_B = np.sum(phi_psi*np.expand_dims(np.expand_dims(G_matrix, axis=-1), axis=-1) , axis=(0,1))
    don_B = (1-parameter['rho']) * np.sum(phi_psi, axis=(0,1))
#    rho_num = np.sum((1-G_matrix)*np.sum(phi_psi, axis=(2,3)))
#    rho_don = np.sum(phi_psi)
#    don_B = (1-rho_num/rho_don) * np.sum(phi_psi, axis=(0,1))
    parameter['B'] = (num_B/don_B).clip(1e-10, 1-1e-10)
    
    
  
    return parameter

def load_graph(PATH):

    with open(os.path.join(PATH, 'network.txt'), 'r') as f:
        network = f.readlines()
    
    G = nx.Graph()
    
    for item in network:
        user1, user2 = item.strip('\n').split(' ')
        G.add_edge(int(user1), int(user2))
        
    G1 = nx.Graph()
    
    for node in G.neighbors(1):
        G1.add_edge(1, node)
        
    for node in G.neighbors(2):
        G1.add_edge(2, node)
        
    for node in G1.nodes():
        if node!=1 and node!=2 and node in G1.neighbors(1) and node in G1.neighbors(2):
            G1.remove_node(node)

#    G1 = nx.Graph()
#    
#    G1.add_edge(1, 2)
#    G1.add_edge(2, 3)
#    G1.add_edge(1, 3)
#    
#    G1.add_edge(4, 5)
#    G1.add_edge(5, 6)
#    G1.add_edge(4, 6)
     
    return G1


if __name__ == '__main__':
    
    PATH = '../twitter/'
    device = 'cpu'
    pretrain_iter = 500
    num_iter = 0
    K = 2
    alpha0 = np.ones(K)
    method = 'mle'
    lr = 1e-6
    inner_lr = 1e-6
    
    print('Loading data...')
    G = load_graph(PATH)
    
    G_matrix = nx.to_numpy_matrix(G)
    G_matrix = np.asarray(G_matrix) + np.eye(len(G_matrix))
    data = {'matrix': G_matrix, 'G': G, 'tweets': [], 'val_tweets': [] }
    
    
    
    parameter = initialize(data, alpha0, K, 1, method, device, lr, inner_lr)
    
    parameter = pretrain(data, parameter, pretrain_iter, lr)
    
    
    
    
    
    
    
    
    
    
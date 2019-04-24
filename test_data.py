#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 21:56:14 2019

@author: yujia
"""

import os

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def plot_seq(tweets):

    num_plot = 10
    for i in range(1,num_plot):
        plt.plot([0,1],[i,i], c = 'k')

#        print(tweets[i])
        plt.scatter(tweets[i], np.ones(len(tweets[i]))*i, c='r')
    plt.xlim([0,1])
    plt.show()

def test_for_tweeter():
    PATH = '../twitter/'
    with open(os.path.join(PATH, 'activity_all.txt'), 'r') as f:
        data = f.readlines()
    
    time_list = []
    tweets = {}
    for item in data:
        time, user = item.strip('\n').split(' ')
        if int(time)>1.48e9 and int(time)<=1.481e9:
            time_list.append(int(time))
            if user in tweets:
                tweets[user].append(time)
            else:
                tweets[user] = [time]
            
    min_time = np.min(time_list)
    max_time = np.max(time_list) - min_time
    
    for key in tweets:
        tweets[key] = [(int(time)-min_time)/max_time for time in tweets[key]]
    
    
    
    
    with open(os.path.join(PATH, 'network.txt'), 'r') as f:
        network = f.readlines()
    
    G = nx.Graph()
    
    for item in network:
        user1, user2 = item.strip('\n').split(' ')
        G.add_edge(int(user1), int(user2))
        
    #match the node number in the two dataset
    #and remove empty sequence 
    
    for key in tweets:
        if not G.has_node(int(key)):
            G.add_node(int(key))
    
    for node in list(G.nodes())[:]:
        if str(node) not in tweets and G.has_node(node):
            G.remove_node(node)
        elif len(tweets[str(node)])<2 and G.has_node(node):
            G.remove_node(node)
            
    ########### make the gragh smaller #############
    #for node in G.nodes():
    #    if node>200:
    #        G.remove_node(node)
    #    nx.draw(G)
    ################################################
    
    tweets_list = []
    val_tweets_list = []
    for node in G.nodes():
        tweets_list.append(tweets[str(node)][:-1])
        val_tweets_list.append(tweets[str(node)][-1])
        
    #%%
    point_list = [time  for tweet in tweets_list for time in tweet] 
    
    plt.hist(point_list, bins=60)
    plt.show()
    
    #%%
    value, interval = np.histogram(point_list, bins=60)
    
    scale = []
    prev = 0
    for item in value:
        new_scale = prev+item
        scale.append(new_scale)
        prev = new_scale
        
    max_scale = scale[-1]
    scale = [item/max_scale for item in scale]
    scale.insert(0,0)
    
    new_tweets_list = []
    for tweet in tweets_list:
        new_tweet = []
        for time in tweet:
            
            index = np.argmax(np.logical_and(interval>time-1./60,interval<time))
            
            new_time = scale[index] + (time-interval[index])*60 *(scale[index+1]-scale[index])
            new_tweet.append(new_time)
        new_tweets_list.append(new_tweet)
        
    plot_seq(tweets_list)
    plot_seq(new_tweets_list)
    
    point_list = [time  for tweet in new_tweets_list for time in tweet] 
    
    plt.hist(point_list, bins=60)
    plt.show()       
    
#test_for_tweeter()
#%%
def test_linkedin_data():
    PATH = '../data_preprocess/PoPPy/data/'  
    
    with open(os.path.join(PATH, 'Linkedin.csv'), 'r') as f:
        data = f.readlines()
        
    seqs = {}
    seqs_org = {}
    max_time = 0
    min_time = 100
    for item in data[1:]:
        user_id, time, org, _ = item.split(',')
        time = float(time)
        if time>max_time:
            max_time = time
        if time<min_time:
            min_time = time
        if user_id in seqs:
            seqs[user_id].append(time)
            seqs_org[user_id].append(org)
        else:
            seqs[user_id] = [time]
            seqs_org[user_id] = [org]
            
    G = nx.Graph()
    
    ## add edge according to be in the same company at the same time
    #for user_i in seqs:
    #    for user_j in seqs:
    #        for p, org_i in enumerate(seqs_org[user_i]):
    #            for q, org_j in enumerate(seqs_org[user_j]):
    #                if org_i == org_j:
    #                    if p==len(seqs_org[user_i])-1 and q==len(seqs_org[user_j])-1:
    #                        G.add_edge(int(user_i), int(user_j))
    #                    elif p==len(seqs_org[user_i])-1:
    #                        if seqs[user_j][q+1]>seqs[user_i][p]:
    #                            G.add_edge(int(user_i), int(user_j))
    #                    elif q==len(seqs_org[user_j])-1:
    #                        if seqs[user_i][p+1]>seqs[user_j][q]:
    #                            G.add_edge(int(user_i), int(user_j))
    #                    elif seqs[user_i][p]>seqs[user_j][q] and seqs[user_i][p]<seqs[user_j][q+1]:
    #                        G.add_edge(int(user_i), int(user_j))
    #                    elif seqs[user_j][q]>seqs[user_i][p] and seqs[user_j][q]<seqs[user_i][p+1]:
    #                        G.add_edge(int(user_i), int(user_j))
                            
    # add edge according to join the same company at the same time
    for user_i in seqs:
        for user_j in seqs:
            for p, org_i in enumerate(seqs_org[user_i]):
                for q, org_j in enumerate(seqs_org[user_j]):
                    if org_i == org_j:
                        if abs(seqs[user_i][p]-seqs[user_j][q])<0.05:
                            G.add_edge(int(user_i), int(user_j))
    
            
    
    #%%
    
    for node in G.nodes():
        if len(seqs[str(node)])<3:
            G.remove_node(node)
            
    print(len(G.nodes()))
    print(len(G.edges()))
    
    linkedin_list = []
    val_linkedin_list = []
    for node in G.nodes():
        if len(seqs[str(node)])>1:
            linkedin_list.append(seqs[str(node)][:-1])
            val_linkedin_list.append(seqs[str(node)][-1])
            
    new_linked_list = []
    new_val_linked_list = []
    time_interval = max_time-min_time
    for i, (time_list, val_time) in enumerate(zip(linkedin_list, val_linkedin_list)):
        new_linked_list.append([(time-min_time)/time_interval for time in time_list])
        new_val_linked_list.append((val_time-min_time)/time_interval)
            
    plot_seq(new_linked_list)              
                    
                    
                    
        
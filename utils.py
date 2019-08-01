#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 12:50:39 2019

@author: yujia
"""
import os

import time
import datetime

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



def load_linkedin_data(PATH):

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
    print('min_time', min_time, 'max_time', max_time)
    G = nx.Graph()

    # add edge according to join the same company at the same time
    for user_i in seqs:
        for user_j in seqs:
            for p, org_i in enumerate(seqs_org[user_i]):
                for q, org_j in enumerate(seqs_org[user_j]):
                    if org_i == org_j:
                        if abs(seqs[user_i][p]-seqs[user_j][q])<0.05:
                            G.add_edge(int(user_i), int(user_j))

    node_to_remove = []
    for node in G.nodes():
        if len(seqs[str(node)])<3:
            node_to_remove.append(node)
    for node in node_to_remove:
        G.remove_node(node)

    print('Number of nodes:', len(G.nodes()))
    print('Number of edges:',len(G.edges()))

    linkedin_list = []
    val_linkedin_list = []
    lenth_list = []
    for node in G.nodes():
        linkedin_list.append(seqs[str(node)][:-1])
        val_linkedin_list.append(seqs[str(node)][-1])
        lenth_list.append(len(seqs[str(node)][:-1]))

    print('Average length:', np.mean(lenth_list))

    new_linked_list = []
    new_val_linked_list = []
    time_interval = max_time-min_time
    for i, (time_list, val_time) in enumerate(zip(linkedin_list, val_linkedin_list)):
        new_linked_list.append([(time-min_time)/time_interval for time in time_list])
        new_val_linked_list.append((val_time-min_time)/time_interval)

#    plot_seq(new_linked_list)

    return G, new_linked_list, new_val_linked_list


def load_overflow_data(PATH, isMath=True):
    G = nx.Graph()
    path = os.path.join(PATH, 'sx-mathoverflow-a2q.txt' if isMath else 'sx-stackoverflow-a2q.txt')

    with open(path, 'r') as f:
        data = f.readlines()

    time_list = []
    user_time_dict = {}
    for item in data:
        user1, user2, time = item.strip('\n').split(' ')
        user1, user2, time = int(user1), int(user2), int(time)

        if isMath:
            time_min = 1.399e9
            time_max = 1.9e9
        else:
            time_min = 1.447e9
            time_max = 1.449e9

        if time >= time_min and time <= time_max:
            time_list.append(time)
            G.add_edge(user1, user2)
            if user1 in user_time_dict:
                user_time_dict[user1].append(time)
            else:
                user_time_dict[user1] = [time]

    min_time = np.min(time_list)
    max_time = np.max(time_list)

    max_time = np.max(time_list) - min_time

    # normalize time and add node to G
    for key, val in user_time_dict.items():
        user_time_dict[key] = [(time-min_time)/max_time for time in val]
        if not G.has_node(key) and (len(val) >= 2 and len(val) <= 10000):
            G.add_node(key)

    # remove nodes in Graph
    node_to_remove = []
    for node in list(G.nodes()):
        if node not in user_time_dict:
            node_to_remove.append(node)
        elif len(user_time_dict[node])  < 2 or len(user_time_dict[node]) > 10000:
            node_to_remove.append(node)
    
    # node_to_remove += nx.isolates(G)
    for node in node_to_remove:
        G.remove_node(node)

    # G.remove_nodes_from(nx.isolates(G))
    node_to_remove = set(list(nx.isolates(G)))
    for node in node_to_remove:
        G.remove_node(node) 
    assert(len(list(nx.isolates(G))) == 0)

    print('Number of nodes:', len(G.nodes()))
    print('Number of edges:', len(G.edges()))


    seq_list = []
    val_seq_list = []
    for key, val in user_time_dict.items():
        if len(val) >= 2 and len(val) <= 10000 and key not in node_to_remove:
            seq_list.append(val[:-1])
            val_seq_list.append(val[-1])
    return G, seq_list, val_seq_list


def load_911_data(PATH):
    G = nx.Graph()
    path = os.path.join(PATH, '911.csv')

    with open(path, 'r') as f:
        data = f.readlines()
    time_list = []
    zip_time_dict = {}

    EMS = 'EMS'
    TRAFFIC = 'TRAFFIC'
    Fire = 'Fire'

    for i in range(len(data)):
        if i > 0:
            _,_,_,zip_code, title,time_str,_,_,_ = data[i].strip('\n').split(',')
            if len(zip_code) == 5 and Fire in title:
                zip_code, timestamp = int(zip_code), time.mktime(datetime.datetime.strptime(time_str, "%m/%d/%y %H:%M").timetuple())
                if timestamp >= 0 and timestamp <= 1.59e9:
                    time_list.append(timestamp)
                    if zip_code in zip_time_dict:
                        zip_time_dict[zip_code].append(timestamp)
                    else:
                        zip_time_dict[zip_code] = [timestamp]
                    


    for zip_code1 in zip_time_dict:
        for zip_code2 in zip_time_dict:
            if (zip_code1 != zip_code2) and (str(zip_code1)[:3] == str(zip_code2)[:3]):
                G.add_edge(zip_code1, zip_code2)

    min_time = np.min(time_list)
    max_time = np.max(time_list)

    abs_max_time = np.max(time_list) - min_time


    # normalize time and add node to G
    for key, val in zip_time_dict.items():
        zip_time_dict[key] = [(time-min_time)/abs_max_time for time in val]
        if not G.has_node(key) and (len(val) >= 2 and len(val) <= 800):
            G.add_node(key)

    # remove nodes in Graph
    node_to_remove = []
    for node in list(G.nodes()):
        if node not in zip_time_dict and G.has_node(node):
            node_to_remove.append(node)
        elif len(zip_time_dict[node])  < 2 or len(zip_time_dict[node]) > 800 and G.has_node(node):
            node_to_remove.append(node)
    for node in node_to_remove:
        G.remove_node(node)

    node_to_remove = set(list(nx.isolates(G)))
    for node in node_to_remove:
        G.remove_node(node) 
    assert(len(list(nx.isolates(G))) == 0)

    print('Number of nodes:', len(G.nodes()))
    print('Number of edges:', len(G.edges()))


    seq_list = []
    val_seq_list = []
    for key, val in zip_time_dict.items():
        if len(val) >= 2 and len(val) <= 800:
            seq_list.append(val[:-1])
            val_seq_list.append(val[-1])
    return G, seq_list, val_seq_list







#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 12:50:39 2019

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

def load_data(PATH):

    with open(os.path.join(PATH, 'activity_all.txt'), 'r') as f:
        data = f.readlines()

    time_list = []
    tweets = {}
    for item in data:
        time, user = item.strip('\n').split(' ')
        if int(time)>1.482e9 and int(time)<=1.483e9:
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
    node_to_remove =[]
    for node in G.nodes():
        if node>200:
            node_to_remove.append(node)
    for node in node_to_remove:
        G.remove_node(node)
#    nx.draw(G)
    ################################################

    tweets_list = []
    val_tweets_list = []
    for node in G.nodes():
        tweets_list.append(tweets[str(node)][:-1])
        val_tweets_list.append(tweets[str(node)][-1])

#    ############### scale the time line #################
#    point_list = [time  for tweet in tweets_list for time in tweet]
#
#    num_bin = 60
#    value, interval = np.histogram(point_list, bins=num_bin,range=(0,1))
#
#    scale = []
#    prev = 0
#    for item in value:
#        new_scale = prev+item
#        scale.append(new_scale)
#        prev = new_scale
#
#    max_scale = scale[-1]
#    scale = [item/max_scale for item in scale]
#    scale.insert(0,0)
#
#    new_tweets_list = []
#    for tweet in tweets_list:
#        new_tweet = []
#        for time in tweet:
#            index = np.argmax(np.logical_and(interval>time-1./num_bin,interval<time))
#            new_time = scale[index] + (time-interval[index])*num_bin *(scale[index+1]-scale[index])
#            new_tweet.append(new_time)
#        new_tweets_list.append(new_tweet)
#
#    new_val_tweets_list = []
#    for time in val_tweets_list:
#        index = np.argmax(np.logical_and(interval>time-1./num_bin,interval<time))
#        new_time = scale[index] + (time-interval[index])*num_bin *(scale[index+1]-scale[index])
#        new_val_tweets_list.append(new_time)
#    ################################################
#    point_list = [time  for tweet in new_tweets_list for time in tweet]
#
#    plt.hist(point_list, bins=num_bin)
#    plt.show()
#
#
#    length = []
#    for tweet in new_tweets_list:
#        length.append(len(tweet))
#    print('mean length', np.mean(length), 'std length', np.std(length))
#
#    return G, new_tweets_list, new_val_tweets_list

    return G, tweets_list, val_tweets_list


def load_linkedin_data(PATH):
#    PATH = '../data_preprocess/PoPPy/data/'

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

    node_to_remove = []
    for node in G.nodes():
        if len(seqs[str(node)])<3:
            node_to_remove.append(node)
    for node in node_to_remove:
        G.remove_node(node)

    ########### make the gragh smaller #############
    for node in G.nodes():
        if node>1000:
            G.remove_node(node)
#    nx.draw(G)
    ################################################

    print(len(G.nodes()))
    print(len(G.edges()))

    linkedin_list = []
    val_linkedin_list = []
    lenth_list = []
    for node in G.nodes():
        linkedin_list.append(seqs[str(node)][:-1])
        val_linkedin_list.append(seqs[str(node)][-1])
        lenth_list.append(len(seqs[str(node)][:-1]))

    print('length', np.mean(lenth_list), np.std(lenth_list))

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
        if isMath or (time >= 1.43e9 and time <= 1.44e9):
            time_list.append(time)
            G.add_edge(user1, user2)
            if user1 in user_time_dict:
                user_time_dict[user1].append(time)
            else:
                user_time_dict[user1] = [time]

    min_time = np.min(time_list)
    max_time = np.max(time_list) - min_time

    print(min_time)
    print(max_time)
    max_time = np.max(time_list) - min_time

    # normalize time and add node to G
    for key, val in user_time_dict.items():
        user_time_dict[key] = [(time-min_time)/max_time for time in val]
        if not G.has_node(key) and (len(val) >= 6 and len(val) <= 7):
            G.add_node(key)

    # remove nodes in Graph
    node_to_remove = []
    for node in list(G.nodes()):
        if node not in user_time_dict and G.has_node(node):
            node_to_remove.append(node)
        elif len(user_time_dict[node])  < 6 or len(user_time_dict[node]) > 7 and G.has_node(node):
            node_to_remove.append(node)
    for node in node_to_remove:
        G.remove_node(node)


    seq_list = []
    val_seq_list = []
    for key, val in user_time_dict.items():
        if len(val) >= 6 and len(val) <= 7:
            seq_list.append(val[:-1])
            val_seq_list.append(val[-1])
    return G, seq_list, val_seq_list


def read_911_data(PATH):
    G = nx.Graph()
    path = os.path.join(PATH, '911.csv')

    with open(path, 'r') as f:
        data = f.readlines()
    time_list = []
    zip_time_dict = {}
    for i in range(len(data)):
        if i:
            _,_,_,zip_code,_,time_str,_,_,_ = data[i].strip('\n').split(',')
            if len(zip_code) == 5:
                zip_code, timestamp = int(zip_code), time.mktime(datetime.datetime.strptime(time_str, "%m/%d/%y %H:%M").timetuple())
                if timestamp >= 1.53e9 and timestamp <= 1.54e9:
                    time_list.append(timestamp)
                    if zip_code in zip_time_dict:
                        zip_time_dict[zip_code].append(timestamp)
                    else:
                        zip_time_dict[zip_code] = [timestamp]

    for zip_code1 in zip_time_dict:
      for zip_code2 in zip_time_dict:
        if (zip_code1 != zip_code2) and (np.abs(zip_code1 - zip_code2) <= 2):
            G.add_edge(zip_code1, zip_code2)

    min_time = np.min(time_list)
    max_time = np.max(time_list)

    print(min_time)
    print(max_time)
    abs_max_time = np.max(time_list) - min_time


    # normalize time and add node to G
    for key, val in zip_time_dict.items():
        zip_time_dict[key] = [(time-min_time)/abs_max_time for time in val]
        if not G.has_node(key) and (len(val) >= 2 and len(val) <= 7):
            G.add_node(key)

    # remove nodes in Graph
    node_to_remove = []
    for node in list(G.nodes()):
        if node not in zip_time_dict and G.has_node(node):
            node_to_remove.append(node)
        elif len(zip_time_dict[node])  < 2 or len(zip_time_dict[node]) > 7 and G.has_node(node):
            node_to_remove.append(node)
    for node in node_to_remove:
        G.remove_node(node)


    seq_list = []
    val_seq_list = []
    for key, val in zip_time_dict.items():
        if len(val) >= 2 and len(val) <= 7:
            seq_list.append(val[:-1])
            val_seq_list.append(val[-1])
    return G, seq_list, val_seq_list



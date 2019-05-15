import networkx as nx
import os, sys
import random
from math import sqrt,floor,ceil
import numpy as np
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly
import plotly.graph_objs as go
import plotly.io as pio
from hawkes_generate import MHP

def get_hawkes_data(mu0,alpha0,omega0,T,var=0.05, max_len = 200):
    while True:
        mu = np.random.normal(mu0,var)
        alpha = np.random.normal(alpha0,var)
        omega = np.random.normal(omega0,var*5)
        if mu < 0.05 or alpha < 0.05 or omega < 0.05 or alpha > 0.97:
            continue
        try:
            P = MHP(mu=np.asarray([mu]), alpha=np.asarray([[alpha]]), omega=omega)
            break
        except:
            continue
    # print(mu,alpha,omega)
    sequence = P.generate_seq(T)[:,0]
    sequence = np.hstack(([0], sequence))
    if len(sequence) > max_len:
        sequence = sequence[:max_len]
    target = sequence[-1]
    sequence = list(sequence)
    sequence.pop()
    return sequence, target, (mu, alpha, omega)

def draw_net(G,clrs,path,filename="network",pos=None,html=True):

        # Get Node Positions
        if pos is None:
            pos=nx.kamada_kawai_layout(G)
        dmin=1
        ncenter=0
        for n in pos:
            x,y=pos[n]
            d=(x-0.5)**2+(y-0.5)**2
            if d<dmin:
                ncenter=n
                dmin=d

        p=nx.single_source_shortest_path_length(G,ncenter)

        # Create Edges
        edge_trace = go.Scatter(
            x=[],
            y=[],
            line=dict(width=2,color='#888'),
            hoverinfo='none',
            mode='lines')

        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])

        node_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            mode='markers',
            hoverinfo='text',
            marker=dict(
                # showscale=True,
                # colorscale options
                #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                # colorscale='YlGnBu',
                # reversescale=True,
                color=[],
                size=40,
                # colorbar=dict(
                #     thickness=15,
                #     title='Node Connections',
                #     xanchor='left',
                #     titleside='right'
                # ),
                line=dict(width=2)))

        for node in G.nodes():
            x, y = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
        # Color Node Points
        for node, adjacencies in enumerate(G.adjacency()):
            clr = clrs[node]
            clr = (round(clr[0]*255),round(clr[1]*255),round(clr[2]*255))
            node_trace['marker']['color'] += tuple(['rgb'+str(clr)]) #tuple([len(adjacencies[1])])
            node_info = 'rgb'+str(clr)
            node_trace['text']+=tuple([node_info])
        # Create Network Graph
        fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    # title='<br>Network graph made with Python',
                    titlefont=dict(size=16),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text="",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
        if not os.path.exists('images'):
            os.mkdir('images')
        if html:
            plotly.offline.plot(fig, filename=os.path.join(path,filename+'.html'), auto_open=False)
        pio.write_image(fig, os.path.join(path,filename+'.pdf'))

def generaldeco(func):
    def wrapper(*args, draw=True, path="./", **kwargs):
        G, seq_train, seq_val, true_param, true_member = func(*args, **kwargs)
        if func.__name__ in ["mmb"]:
            dist = {}
            for node1 in list(G.nodes):
                dist[node1] = {}
                for node2 in list(G.nodes):
                    if node1 == node2:
                        continue
                    if true_member[node1] == true_member[node2]:
                        dist[node1][node2] = 0.6
                    else:
                        dist[node1][node2] = 1
            pos=nx.kamada_kawai_layout(G, dist = dist)
            # pos=nx.kamada_kawai_layout(G)
        else:
            pos=nx.kamada_kawai_layout(G)
        if draw:
            import copy
            clrs = np.array(copy.copy(true_param))
            clrs[:,2] /= 10
            draw_net(G,clrs,path,pos=pos)

            num_cluster = max(true_member)+1
            if num_cluster > 10:
                print("Warning! Don't have enough colors, random generate color map")
                color_map = [(random.random(),random.random(),random.random()) for i in range(num_cluster)]
            else:
                color_map = plt.get_cmap('tab10').colors[:num_cluster]
            color_map = np.array(color_map)
            clrs = color_map[true_member,:]
            draw_net(G,clrs,path,filename="true_membership",pos=pos,html=False)
        return G, seq_train, seq_val, true_param, true_member, pos
    return wrapper

@generaldeco
def balanced_tree(r=5,h=3,style=0,h_thr=None,T=100):
    """
    r: number of rays
    h: depth of tree
    style: 0 -- cluster based on h_thr
    """
    if style==0:
        if h_thr is None:
            h_thr = list(range(h))
        elif h_thr.__class__ is int:
            h_thr = [h_thr]
    G = nx.Graph()
    seq_train = []
    seq_val = []
    true_param = []
    true_member = []
    member_id = 0

    node_id = 0
    G.add_node(0)
    layer = [0]
    mu = random.uniform(0.15,0.85)
    alpha = random.uniform(0.15,0.85)
    omega = random.uniform(1,10)
    sequence, target, param = get_hawkes_data(mu,alpha,omega,T)
    seq_train += [sequence]
    seq_val += [target]
    true_param += [param]
    true_member += [member_id]
    for i in range(h):
        new_layer = []
        if style==0 and (i in h_thr):
            mu = random.uniform(0.15,0.85)
            alpha = random.uniform(0.15,0.85)
            omega = random.uniform(1,10)
            member_id += 1
        for rootnode in layer:
            for j in range(r):
                node_id += 1
                G.add_node(node_id)
                G.add_edge(rootnode,node_id)
                new_layer += [node_id]
                sequence, target, param = get_hawkes_data(mu,alpha,omega,T)
                seq_train += [sequence]
                seq_val += [target]
                true_param += [param]
                true_member += [member_id]
        layer = new_layer
    return G, seq_train, seq_val, true_param, true_member

@generaldeco
def balanced_treev2(tr=3,r=5,h=3,T=100):
    """
    tr: number of subtrees
    r: number of rays
    h: depth of tree
    style: 0 -- cluster based on h_thr
    """
    G = nx.Graph()
    seq_train = []
    seq_val = []
    true_param = []
    true_member = []

    member_id = -1
    node_id = -1

    tr_nodes=[]
    for t in range(tr):
        node_id += 1
        member_id += 1
        G.add_node(node_id)
        for pr_tr in tr_nodes:
            G.add_edge(pr_tr,node_id)
        tr_nodes += [node_id]

        layer = [node_id]
        mu = random.uniform(0.15,0.85)
        alpha = random.uniform(0.15,0.85)
        omega = random.uniform(1,10)
        sequence, target, param = get_hawkes_data(mu,alpha,omega,T)
        seq_train += [sequence]
        seq_val += [target]
        true_param += [param]
        true_member += [member_id]
        for i in range(h):
            new_layer = []
            for rootnode in layer:
                for j in range(r):
                    node_id += 1
                    G.add_node(node_id)
                    G.add_edge(rootnode,node_id)
                    new_layer += [node_id]
                    sequence, target, param = get_hawkes_data(mu,alpha,omega,T)
                    seq_train += [sequence]
                    seq_val += [target]
                    true_param += [param]
                    true_member += [member_id]
            layer = new_layer
    return G, seq_train, seq_val, true_param, true_member


@generaldeco
def barbell(tr=3,m1=5,m2=2,T=100):
    """
    tr: number of subtrees
    r: number of rays
    h: depth of tree
    style: 0 -- cluster based on h_thr
    """
    G = nx.Graph()
    seq_train = []
    seq_val = []
    true_param = []
    true_member = []

    member_id = -1
    node_id = -1

    tr_nodes=[]
    for t in range(tr):
        node_id += 1
        member_id += 1
        G.add_node(node_id)
        for pr_tr in tr_nodes:
            G.add_edge(pr_tr,node_id)
        tr_nodes += [node_id]

        mu = random.uniform(0.15,0.85)
        alpha = random.uniform(0.15,0.85)
        omega = random.uniform(1,10)
        sequence, target, param = get_hawkes_data(mu,alpha,omega,T)
        seq_train += [sequence]
        seq_val += [target]
        true_param += [param]
        true_member += [member_id]





        for i in range(m2):
            node_id += 1
            G.add_node(node_id)
            G.add_edge(node_id-1,node_id)
            sequence, target, param = get_hawkes_data(mu,alpha,omega,T)
            seq_train += [sequence]
            seq_val += [target]
            true_param += [param]
            true_member += [member_id]
        tr_root = node_id
        for i in range(m1):
            node_id += 1
            G.add_node(node_id)
            for j in range(tr_root,node_id):
                G.add_edge(j,node_id)
            sequence, target, param = get_hawkes_data(mu,alpha,omega,T)
            seq_train += [sequence]
            seq_val += [target]
            true_param += [param]
            true_member += [member_id]

    return G, seq_train, seq_val, true_param, true_member

@generaldeco
def mmb(nodes=20,clusters=3,bjk=1.5,bkk=10,hardedge=False,T=100):
    """
    nodes: number of nodes
    clusters: number of clusters
    hardedge: true use true membership to draw the edges
              false use mixed_membership resample member to draw the edges
    """
    G = nx.Graph()
    seq_train = []
    seq_val = []
    true_param = []
    true_member = []

    cluster_param = []
    for k in range(clusters):
        mu = random.uniform(0.15,0.85)
        alpha = random.uniform(0.15,0.85)
        omega = random.uniform(1,10)
        cluster_param += [(mu,alpha,omega)]

    # alpha
    alpha0 = np.random.dirichlet(np.ones(clusters))+1

    mixed_membership = [] # pi
    for node_id in range(nodes):
        G.add_node(node_id)
        mixed_membership += [np.random.dirichlet(alpha0)]
        member_id = np.random.multinomial(1,mixed_membership[-1]).argmax()
        true_member += [member_id]

        mu,alpha,omega = cluster_param[member_id]
        sequence, target, param = get_hawkes_data(mu,alpha,omega,T)
        seq_train += [sequence]
        seq_val += [target]
        true_param += [param]

    # B
    B = np.eye(clusters)
    if hardedge:
        populations = [sum([j==i for j in true_member]) for i in range(clusters)]
    else:
        populations = alpha0/sum(alpha0)*nodes
    for i in range(clusters):
        for j in range(clusters):
            if i==j:
                continue
            B[i,j] = bjk/nodes
    B = (B.T+B)/2
    for i in range(clusters):
        B[i,i] = bkk /populations[i]
        if B[i,i]>1:
            print("Warning! Increase number of nodes.")
            B[i,i]=0.95
    for node_id1 in range(nodes):
        for node_id2 in range(node_id1+1,nodes):
            if hardedge:
                z1 = true_member[node_id1]
                z2 = true_member[node_id2]
            else:
                z1 = np.random.multinomial(1,mixed_membership[node_id1]).argmax()
                z2 = np.random.multinomial(1,mixed_membership[node_id2]).argmax()
            if np.random.binomial(1,B[z1,z2]):
                G.add_edge(node_id1,node_id2)

    return G, seq_train, seq_val, true_param, true_member




















#-----------------------------------------------------------------------#
# Package: High-dimensional Undirected Graph Estimation                 #
# huge.generator(): Data generator                                      #
#-----------------------------------------------------------------------#

#' Data generator
#'
#' Implements the data generation from multivariate normal distributions with different graph structures, including \code{"random"}, \code{"hub"}, \code{"cluster"}, \code{"band"} and \code{"scale-free"}.
#'
#' @param n The number of observations (sample size). The default value is \code{200}.
#' @param d The number of variables (dimension). The default value is \code{50}.
#' @param graph The graph structure with 4 options: \code{"random"}, \code{"hub"}, \code{"cluster"}, \code{"band"} and \code{"scale-free"}.
#' @param v The off-diagonal elements of the precision matrix, controlling the magnitude of partial correlations with \code{u}. The default value is \code{0.3}.
#' @param u A positive number being added to the diagonal elements of the precision matrix, to control the magnitude of partial correlations. The default value is \code{0.1}.
#' @param g For \code{"cluster"} or \code{"hub"} graph, \code{g} is the number of hubs or clusters in the graph. The default value is about \code{d/20} if \code{d >= 40} and \code{2} if \code{d < 40}. For \code{"band"} graph, \code{g} is the bandwidth and the default value is \code{1}. NOT applicable to \code{"random"} graph.
#' @param prob For \code{"random"} graph, it is the probability that a pair of nodes has an edge. The default value is \code{3/d}. For \code{"cluster"} graph, it is the probability that a pair of nodes has an edge in each cluster. The default value is \code{6*g/d} if \code{d/g <= 30} and \code{0.3} if \code{d/g > 30}. NOT applicable to \code{"hub"} or \code{"band"} graphs.
#' @param vis Visualize the adjacency matrix of the True graph structure, the graph pattern, the covariance matrix and the empirical covariance matrix. The default value is \code{False}
#' @param verbose If \code{verbose = False}, tracing information printing is disabled. The default value is \code{True}.
#' @details
#' Given the adjacency matrix \code{theta}, the graph patterns are generated as below:\cr\cr
#' (I) \code{"random"}: Each pair of off-diagonal elements are randomly set \code{theta[i,j]=theta[j,i]=1} for \code{i!=j} with probability \code{prob}, and \code{0} other wise. It results in about \code{d*(d-1)*prob/2} edges in the graph.\cr\cr
#' (II)\code{"hub"}:The row/columns are evenly partitioned into \code{g} disjoint groups. Each group is associated with a "center" row \code{i} in that group. Each pair of off-diagonal elements are set \code{theta[i,j]=theta[j,i]=1} for \code{i!=j} if \code{j} also belongs to the same group as \code{i} and \code{0} otherwise. It results in \code{d - g} edges in the graph.\cr\cr
#' (III)\code{"cluster"}:The row/columns are evenly partitioned into \code{g} disjoint groups. Each pair of off-diagonal elements are set \code{theta[i,j]=theta[j,i]=1} for \code{i!=j} with the probability \code{prob}if both \code{i} and \code{j} belong to the same group, and \code{0} other wise. It results in about \code{g*(d/g)*(d/g-1)*prob/2} edges in the graph.\cr\cr
#' (IV)\code{"band"}: The off-diagonal elements are set to be \code{theta[i,j]=1} if \code{1<=|i-j|<=g} and \code{0} other wise. It results in \code{(2d-1-g)*g/2} edges in the graph.\cr\cr
#' (V) \code{"scale-free"}: The graph is generated using B-A algorithm. The initial graph has two connected nodes and each new node is connected to only one node in the existing graph with the probability proportional to the degree of the each node in the existing graph. It results in \code{d} edges in the graph.
#'
#' The adjacency matrix \code{theta} has all diagonal elements equal to \code{0}. To obtain a positive definite precision matrix, the smallest eigenvalue of \code{theta*v} (denoted by \code{e}) is computed. Then we set the precision matrix equal to \code{theta*v+(|e|+0.1+u)I}. The covariance matrix is then computed to generate multivariate normal data.
#' @return
#' An object with S3 class "sim" is returned:
#' \item{data}{
#'   The \code{n} by \code{d} matrix for the generated data
#' }
#' \item{sigma}{
#'   The covariance matrix for the generated data
#' }
#' \item{omega}{
#'   The precision matrix for the generated data
#' }
#' \item{sigmahat}{
#'   The empirical covariance matrix for the generated data
#' }
#' \item{theta}{
#'   The adjacency matrix of True graph structure (in sparse matrix representation) for the generated data
#' }
#' @seealso \code{\link{huge}} and \code{\link{huge-package}}
#' @examples
#' ## band graph with bandwidth 3
#' L = huge.generator(graph = "band", g = 3)
#' plot(L)
#'
#' ## random sparse graph
#' L = huge.generator(vis = True)
#'
#' ## random dense graph
#' L = huge.generator(prob = 0.5, vis = True)
#'
#' ## hub graph with 6 hubs
#' L = huge.generator(graph = "hub", g = 6, vis = True)
#'
#' ## hub graph with 8 clusters
#' L = huge.generator(graph = "cluster", g = 8, vis = True)
#'
#' ## scale-free graphs
#' L = huge.generator(graph="scale-free", vis = True)
#' @export
# def generate(n = 200, d = 50, graph = "random", v = None, u = None, g = None, prob = None, vis = False, verbose = True){
#   if(verbose) print("Generating data from the multivariate normal distribution with the", graph,"graph structure....")
#   if(g == None){
#     g = 1
#     if(graph == "hub" or graph == "cluster"){
#       if(d > 40)  g = ceil(d//20)
#       if(d <= 40) g = 2
#     }
#   }
#
#   if(graph == "random"){
#     if(prob == None)  prob = min(1, 3/d)
#     prob = sqrt(prob/2)*(prob<0.5)+(1-sqrt(0.5-0.5*prob))*(prob>=0.5)
#   }
#
#   if(graph == "cluster"){
#     if(prob == None){
#       if(d/g > 30)  prob = 0.3
#       if(d/g <= 30)  prob = min(1,6*g/d)
#     }
#     prob = sqrt(prob/2)*(prob<0.5)+(1-sqrt(0.5-0.5*prob))*(prob>=0.5)
#   }
#
#
#   # parition variables into groups
#   g_large = d%g
#   g_small = g - g_large
#   n_small = floor(d/g)
#   n_large = n_small+1
#   g_list = np.array([n_small]*g_small + [n_large]*g_large)
#   #g_list = c(rep(n_small,g_small),rep(n_large,g_large))
#   g_ind = np.array([n_small]*g_small + [n_large]*g_large)
#   g_ind = rep(c(1:g),g_list)
#   rm(g_large,g_small,n_small,n_large,g_list)
#   gc()
#
#   # build the graph structure
#   theta = matrix(0,d,d);
#   if(graph == "band"){
#     if(u == None) u = 0.1
#     if(v == None) v = 0.3
#     for(i in 1:g){
#       diag(theta[1:(d-i),(1+i):d]) = 1
#       diag(theta[(1+i):d,1:(d-1)]) = 1
#     }
#   }
#   if(graph == "cluster"){
#     if(u==None) u = 0.1
#     if(v==None) v = 0.3
#     for(i in 1:g){
#        tmp = which(g_ind==i)
#        tmp2 = matrix(runif(length(tmp)^2,0,0.5),length(tmp),length(tmp))
#        tmp2 = tmp2 + t(tmp2)
#        theta[tmp,tmp][tmp2<prob] = 1
#        rm(tmp,tmp2)
#        gc()
#     }
#   }
#   if(graph == "hub"){
#     if(u == None) u = 0.1
#     if(v == None) v = 0.3
#     for(i in 1:g){
#        tmp = which(g_ind==i)
#        theta[tmp[1],tmp] = 1
#        theta[tmp,tmp[1]] = 1
#        rm(tmp)
#        gc()
#     }
#   }
#   if(graph == "random"){
#     if(u == None) u = 0.1
#     if(v == None) v = 0.3
#
#     tmp = matrix(runif(d^2,0,0.5),d,d)
#     tmp = tmp + t(tmp)
#     theta[tmp < prob] = 1
#     #theta[tmp >= tprob] = 0
#     rm(tmp)
#     gc()
#   }
#
#   if(graph == "scale-free"){
#   if(u == None) u = 0.1
#   if(v == None) v = 0.3
#   out = .Call("_huge_SFGen", 2, d)
#   theta = matrix(as.numeric(out$G),d,d)
#   }
#   diag(theta) = 0
#   omega = theta*v
#
#   # make omega positive definite and standardized
#   diag(omega) = abs(min(eigen(omega)$values)) + 0.1 + u
#   sigma = cov2cor(solve(omega))
#   omega = solve(sigma)
#
#   # generate multivariate normal data
#   x = mvrnorm(n,rep(0,d),sigma)
#   sigmahat = cor(x)
#
#   # graph and covariance visulization
#   if(vis == True){
#   fullfig = par(mfrow = c(2, 2), pty = "s", omi=c(0.3,0.3,0.3,0.3), mai = c(0.3,0.3,0.3,0.3))
#   fullfig[1] = image(theta, col = gray.colors(256),  main = "Adjacency Matrix")
#
#   fullfig[2] = image(sigma, col = gray.colors(256), main = "Covariance Matrix")
#   g = graph.adjacency(theta, mode="undirected", diag=False)
#   layout.grid = layout.fruchterman.reingold(g)
#
#   fullfig[3] = plot(g, layout=layout.grid, edge.color='gray50',vertex.color="red", vertex.size=3, vertex.label=NA,main = "Graph Pattern")
#
#   fullfig[4] = image(sigmahat, col = gray.colors(256), main = "Empirical Matrix")
#   rm(fullfig,g,layout.grid)
#   gc()
#   }
#   if(verbose) print("done.\n")
#   rm(vis,verbose)
#   gc()
#
#   sim = list(data = x, sigma = sigma, sigmahat = sigmahat, omega = omega, theta = Matrix(theta,sparse = True), sparsity= sum(theta)/(d*(d-1)), graph.type=graph)
#   class(sim) = "sim"
#   return(sim)
# }
#
# #' Print function for S3 class "sim"
# #'
# #' Print the information about the sample size, the dimension, the pattern and sparsity of the True graph structure.
# #'
# #' @param x An object with S3 class \code{"sim"}.
# #' @param \dots System reserved (No specific usage)
# #' @seealso \code{\link{huge.generator}}
# #' @export
# print.sim = function(x, ...){
#   print("Simulated data generated by huge.generator()\n")
#   print("Sample size: n =", nrow(x$data), "\n")
#   print("Dimension: d =", ncol(x$data), "\n")
#     print("Graph type = ", x$graph.type, "\n")
#     print("Sparsity level:", sum(x$theta)/ncol(x$data)/(ncol(x$data)-1),"\n")
# }
#
# #' Plot function for S3 class "sim"
# #'
# #' Visualize the covariance matrix, the empirical covariance matrix, the adjacency matrix and the graph pattern of the True graph structure.
# #'
# #' @param x An object with S3 class \code{"sim"}
# #' @param \dots System reserved (No specific usage)
# #' @seealso \code{\link{huge.generator}} and \code{\link{huge}}
# #' @export
# plot.sim = function(x, ...){
#   gcinfo(False)
#      par = par(mfrow = c(2, 2), pty = "s", omi=c(0.3,0.3,0.3,0.3), mai = c(0.3,0.3,0.3,0.3))
#      image(as.matrix(x$theta), col = gray.colors(256),  main = "Adjacency Matrix")
#   image(x$sigma, col = gray.colors(256), main = "Covariance Matrix")
#   g = graph.adjacency(x$theta, mode="undirected", diag=False)
#   layout.grid = layout.fruchterman.reingold(g)
#
#   plot(g, layout=layout.grid, edge.color='gray50',vertex.color="red", vertex.size=3, vertex.label=NA,main = "Graph Pattern")
#   rm(g, layout.grid)
#   gc()
#   image(x$sigmahat, col = gray.colors(256), main = "Empirical Covariance Matrix")
# }

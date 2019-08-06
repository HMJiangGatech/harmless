#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 16:35:33 2019

@author: yujia
"""
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc
import numpy as np

class Hawkes_univariant(nn.Module):
    def __init__(self, init_param, T, inner_lr=1e-6, device='cpu'):
        super(Hawkes_univariant, self).__init__()

        self.mu = torch.randn(1, device=device, requires_grad=True)
        self.alpha = torch.randn(1, device=device, requires_grad=True)
        self.w = torch.randn(1, device=device, requires_grad=True)

        self.mu.data.fill_(init_param[0])
        self.alpha.data.fill_(init_param[1])
        self.w.data.fill_(init_param[2])

        self.reg = 1e-2
        self.T = T
        self.inner_lr = inner_lr

    def evaluate(self, x, pred, update_mu, update_alpha, update_w):
        delta = pred-x
        delta = torch.exp(-update_w*delta)
        delta = update_alpha * update_w * torch.sum(delta)

        pos = torch.log(update_mu + delta)
        neg = update_alpha * (1- torch.exp(-update_w*(pred-x[:,-1])))
        NLL = update_mu * (pred-x[:,-1]) - pos + neg

        return NLL
    def get_prob(self, x, delta_T, update_mu, update_alpha, update_w):
        delta = x[:,-1] - x
        delta = torch.exp(-update_w*delta)
        delta = update_alpha * (torch.exp(-update_w*delta_T)-1) * torch.sum(delta)

        prob = 1 - torch.exp(-update_mu*delta_T + delta)
#        print(prob)
        return prob

    def update_once(self, x):
        raise NotImplementedError

    def compute_loss(self, x, update_mu, update_alpha, update_w):

        delta = x.unsqueeze(1)-x.unsqueeze(2)
        w_t_delta = -update_w*delta
        w_t_delta =  w_t_delta.clone().masked_fill_(delta<=0, -float('inf'))
        delta1 = torch.exp(w_t_delta)
        delta3 = update_alpha * update_w * torch.sum(delta1, dim=1)

        pos = torch.log(update_mu + delta3)
        neg = update_alpha * (1- torch.exp(-update_w*(self.T-x)))
        NLL = update_mu * self.T - torch.sum(pos) +torch.sum(neg)

        loss = NLL+ self.reg* (-torch.log(update_mu)-torch.log(update_alpha)-torch.log(update_w))

        return loss

    def donot_update_once(self, x):
        return self.mu, self.alpha, self.w

    def forward(self, x):

        update_mu, update_alpha, update_w = self.update_once(x)
        return self.compute_loss(x, update_mu, update_alpha, update_w)

class Hawkes_mle(Hawkes_univariant):
    def __init__(self, init_param, T, inner_lr=1e-6, device='cpu'):
        super(Hawkes_mle, self).__init__(init_param, T, inner_lr=1e-6, device='cpu')

    def update_once(self, x):
        return self.mu, self.alpha, self.w


class Hawkes_maml(Hawkes_univariant):
    def __init__(self, init_param, T, inner_lr=1e-6, device='cpu'):
        super(Hawkes_maml, self).__init__(init_param, T, inner_lr=inner_lr, device=device)


    def update_once(self, x):


        delta = x.unsqueeze(1)-x.unsqueeze(2)
        w_t_delta = -self.w*delta
        w_t_delta =  w_t_delta.clone().masked_fill_(delta<=0, -float('inf'))
        delta1 = torch.exp(w_t_delta)
        delta2 = torch.sum(delta1, dim=1)

        inv_log = 1./(self.mu + self.alpha * self.w * delta2)

        deltatti = self.T-x
        deltatti2 = -self.w*deltatti
        deltatti3 = torch.exp(deltatti2)



        delta_mu = -(-self.T + inv_log.sum()) #- 1./self.mu
        update_mu =  self.mu - self.inner_lr*delta_mu

        delta_alpha = -(-len(x) + torch.sum(deltatti3) + (inv_log*self.w*delta2).sum()) #-1./self.alpha
        update_alpha = self.alpha - self.inner_lr*delta_alpha

        w_t_delta_t_delta =  (w_t_delta*delta1).clone().masked_fill_(delta<=0, 0.)

        delta_w = -(-self.alpha*torch.sum(deltatti3*deltatti) \
                  + (inv_log*self.alpha*(delta2 + torch.sum(w_t_delta_t_delta,dim=1))).sum()) #-1./self.w

        update_w = self.w - self.inner_lr*delta_w

        if update_mu<0:
            update_mu = -update_mu
        if update_alpha<0:
            update_alpha = -update_alpha
        if update_w<0:
            update_w = -update_w
        return update_mu, update_alpha, update_w





class Hawkes_models():
    def __init__(self, data, T, K, method, lr=1e-2, inner_lr=1e-2, device='cpu', Tensor = torch.FloatTensor, init='random'):

        self.method = method

        if self.method == 'mle':

            self.models = []
            if init == 'random':
                theta = np.exp(np.random.normal(size=(K, 3)))*0.1
            if init == 'uniform':
                theta = np.zeros(shape=(K, 3))+0.1
            for k in range(K):
                self.models.append(Hawkes_mle(list(theta[k,:]),T, device=device, inner_lr=inner_lr))

            params = [[model.mu, model.alpha, model.w] for model in self.models]
            params = [item for sublist in params for item in sublist]
            self.optimizer = torch.optim.SGD(params, lr=lr)


        elif self.method == 'maml':
            self.models = []
            if init == 'random':
                theta = np.exp(abs(np.random.normal(size=(K, 3))))*0.1#+0.1
            if init == 'uniform':
                theta = np.zeros(shape=(K, 3))+0.1
            for k in range(K):
                self.models.append(Hawkes_maml(list(theta[k,:]),T, device=device,inner_lr=inner_lr))

            params = [[model.mu, model.alpha, model.w] for model in self.models]
            params = [item for sublist in params for item in sublist]
            self.optimizer = torch.optim.SGD(params, lr=lr)

        elif self.method =='fomaml' or self.method == 'reptile':
            self.models = []
            if init == 'random':
                theta = np.exp(np.random.normal(size=(K, 3)))*0.1
            if init == 'uniform':
                theta = np.zeros(shape=(K, 3))+0.1
            for k in range(K):
                self.models.append(Hawkes_mle(list(theta[k,:]),T, device=device))

            params = [[model.mu, model.alpha, model.w] for model in self.models]
            params = [item for sublist in params for item in sublist]
            self.optimizer = torch.optim.SGD(params, lr=lr)

            if init == 'random':
                theta = np.exp(np.random.normal(size=(3)))*0.1
            if init == 'uniform':
                theta = np.zeros(shape=(3))+0.1
#            theta = np.zeros(shape=(3))+0.1
            self.shadow_model = Hawkes_mle(list(theta),T, device=device)

            self.shadow_params = [self.shadow_model.mu, self.shadow_model.alpha, self.shadow_model.w]
            self.shadow_optimizer = torch.optim.SGD(self.shadow_params, lr=inner_lr)

            # for evaluation
            if init == 'random':
                theta = np.exp(np.random.normal(size=(3)))*0.1
            if init == 'uniform':
                theta = np.zeros(shape=(3))+0.1
            self.model_tester = Hawkes_mle(list(theta),T, device=device)

            self.params_tester = [self.model_tester.mu, self.model_tester.alpha, self.model_tester.w]
            self.optimizer_tester = torch.optim.SGD(self.params_tester, lr=inner_lr)
        else:
            raise NotImplementedError

#        for tweet in data['tweets']:
#            if len(tweet)==0:
#                tweet.append(0)
        self.tweets = [Tensor(item).unsqueeze(0) for item in data['tweets']]
        self.val_tweets = data['val_tweets']
        self.K = K
        self.N = len(self.tweets)
        self.tensor = Tensor
        self.L = None
        self.weights = None
        self.updates = None
        self.inner_lr = inner_lr
        self.lr = lr

    def compute_loss(self):

        if self.method == 'mle' or self.method == 'maml':
            self.L = torch.zeros(self.N, self.K)
            for user_id, tweet in enumerate(self.tweets):
#                if len(tweet)==0: continue
                for k, model in enumerate(self.models):
                    # print(tweet.shape)
                    self.L[user_id, k] = model(tweet)
            return self.L.data.numpy()


        elif self.method == 'fomaml':
            self.L = np.zeros((self.N, self.K))
            self.weights = [[] for _ in range(self.N)]
            for user_id, tweet in enumerate(self.tweets):
                for k, model in enumerate(self.models):
                    loss = model(tweet)
                    params = [model.mu, model.alpha, model.w]
                    grad = torch.autograd.grad(loss, params)
                    self.weights[user_id].append(list(map(lambda p: p[1] - self.inner_lr * p[0], zip(grad, params))))
                    self.L[user_id, k] = loss.data.item()
            return self.L
        elif self.method == 'reptile':
            self.L = np.zeros((self.N, self.K))
            self.updates = [[[] for _ in range(self.K)] for _ in range(self.N)]

            for user_id, tweet in enumerate(self.tweets):
                for k, model in enumerate(self.models):
                    params = [model.mu, model.alpha, model.w]

                    for param1, param2 in zip(params,self.shadow_params):
                        param2.data = param1.data.clone()
                    for _ in range(1):
                        self.shadow_optimizer.zero_grad()
                        loss = self.shadow_model(tweet)
                        loss.backward()
                        self.L[user_id, k] = loss.data.item()
                        self.shadow_optimizer.step()


                    for param1, param2 in zip(self.shadow_params,params):
                        self.updates[user_id][k].append(param2.data-param1.data)

            return self.L
        else:
            raise NotImplementedError

    def project(self):
        for k, model in enumerate(self.models):
            for s in (model.mu,model.alpha,model.w):
                if s.data < 0:
                    s.data = -s.data

    def update_theta(self, loss_weights):
        if self.L is None:
            print('please run compute_loss fist!')

        if self.method == 'mle' or self.method == 'maml':
            self.optimizer.zero_grad()

            loss = torch.sum(self.L*self.tensor(loss_weights))

            loss.backward()
            self.optimizer.step()
            self.project()
            return loss.data.item()
        elif self.method == 'fomaml':
            loss_accum = 0
            for user_id, tweet in enumerate(self.tweets):
                # print(user_id)
                self.optimizer.zero_grad()
                for k, (model, weight) in enumerate(zip(self.models,self.weights[user_id])):
                    params = [model.mu, model.alpha, model.w]
                    # print(wessight)
                    for param1, param2 in zip(weight,self.shadow_params):
                        param2.data = param1.data.clone()

                    loss = loss_weights[user_id,k] * self.shadow_model(tweet)
                    loss.backward()
                    loss_accum += loss.data.item()
                    for param1, param2 in zip(self.shadow_params,params):
                        try:
                            param2.grad.data.add_(param1.grad)
                        except:
                            param2.grad = param1.grad
                self.optimizer.step()

            return loss_accum

        elif self.method == 'reptile':
            loss_accum = 0
            for user_id, tweet in enumerate(self.tweets):
                # print(user_id)
                self.optimizer.zero_grad()
                for k, (model, update) in enumerate(zip(self.models,self.updates[user_id])):
                    params = [model.mu, model.alpha, model.w]
                    # print(update)
                    for param1, param2 in zip(update,params):
                        if param2.grad is None:
                            param2.grad = loss_weights[user_id,k] * param1
                        else:
                            param2.grad.data.add_(loss_weights[user_id,k] * param1)
                self.optimizer.step()

            return np.sum(self.L*loss_weights)
        else:
            raise NotImplementedError

    def train(self):
        for model in self.models:
            model.train()

    def eval(self):
        for model in self.models:
            model.eval()

    def show_param(self):
        for k, model in enumerate(self.models):
            print('Model:', k, "mu:",model.mu.item(),"alpha:",model.alpha.item(),"w:",model.w.item())

    def get_eval_param(self, model, seq):
        if self.method == 'mle' or self.method == 'maml':
            return model.update_once(seq)
        elif self.method == 'fomaml' or self.method == 'reptile':


            params = [model.mu, model.alpha, model.w]
            for param1, param2 in zip(self.params_tester,params):
                param1.data = param2.data.clone()
            for _ in range(1):
            #update once
                self.optimizer_tester.zero_grad()
                loss = self.model_tester(seq)
                loss.backward()
                self.optimizer_tester.step()

            return self.model_tester.mu, self.model_tester.alpha, self.model_tester.w

    def evaluate(self, weights):
        accum_nll = 0
        for model in self.models:
            model.eval()
        for i, (seq, target) in enumerate(zip(self.tweets,self.val_tweets)):
            accum_likelihood = 0
            for k, model in enumerate(self.models):
                #update once
                update_mu, update_alpha, update_w = self.get_eval_param(model, seq)
                #evaluate
                nll = model.evaluate(seq, target, update_mu, update_alpha, update_w)
                accum_likelihood += weights[i,k]*np.exp(-nll.data.item())
            accum_nll += -np.log(accum_likelihood)

        return accum_nll/self.N


    def get_roc_auc(self, weights, delta_T):
        self.prob_list = []
        self.truth_list = []

        for i, (seq, target) in enumerate(zip(self.tweets,self.val_tweets)):
            if bool(target-seq[0][-1]<=delta_T):
                self.truth_list.append(1)
            else:
                self.truth_list.append(0)
            mix_prob = 0
            for k, model in enumerate(self.models):
                update_mu, update_alpha, update_w = self.get_eval_param(model, seq)
                prob = model.get_prob(seq, delta_T, update_mu, update_alpha, update_w)
                mix_prob += weights[i,k]*prob.data.item()
            self.prob_list.append(mix_prob)

        fpr, tpr, _ = roc_curve(self.truth_list, self.prob_list)

        roc_auc = auc(fpr, tpr)

        return roc_auc, fpr, tpr

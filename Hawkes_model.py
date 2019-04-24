#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 16:35:33 2019

@author: yujia
"""
import torch
import torch.nn as nn

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

        self.reg = 1e-10
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
        super(Hawkes_maml, self).__init__(init_param, T, inner_lr=1e-6, device='cpu')    
    
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
        
        
        
        delta_mu = -self.T + inv_log.sum() - 1./self.mu
        update_mu =  self.mu - self.inner_lr*delta_mu
        
        delta_alpha = -len(x) + torch.sum(deltatti3) + (inv_log*self.w*delta2).sum() -1./self.alpha
        update_alpha = self.alpha - self.inner_lr*delta_alpha
        
        w_t_delta_t_delta =  (w_t_delta*delta1).clone().masked_fill_(delta<=0, 0.)
        
        delta_w = -self.alpha*torch.sum(deltatti3*deltatti) \
                  + (inv_log*self.alpha*(delta2 + torch.sum(w_t_delta_t_delta,dim=1))).sum() -1./self.w

        update_w = self.w - self.inner_lr*delta_w


        return update_mu, update_alpha, update_w

            

    
    
class Hawkes_models():
    def __init__(self, data, T, K, method, lr=1e-2, inner_lr=1e-2, device='cpu', Tensor = torch.FloatTensor):
        
        self.method = method
        
        if self.method == 'mle':
        
            self.models = []
            theta = np.exp(np.random.normal(size=(K, 3)))*0.1
            for k in range(K):
                self.models.append(Hawkes_mle(list(theta[k,:]),T, device=device))
            
            params = [[model.mu, model.alpha, model.w] for model in self.models]
            params = [item for sublist in params for item in sublist]
            self.optimizer = torch.optim.SGD(params, lr=lr)
            
            
        elif self.method == 'maml':
            self.models = []
            theta = np.exp(np.random.normal(size=(K, 3)))*0.1
            for k in range(K):
                self.models.append(Hawkes_maml(list(theta[k,:]),T, device=device))
            
            params = [[model.mu, model.alpha, model.w] for model in self.models]
            params = [item for sublist in params for item in sublist]
            self.optimizer = torch.optim.SGD(params, lr=lr)
            
        elif self.method =='fomaml' or self.method == 'reptile':
            self.models = []
            theta = np.exp(np.random.normal(size=(K, 3)))*0.1
            for k in range(K):
                self.models.append(Hawkes_mle(list(theta[k,:]),T, device=device))
            
            params = [[model.mu, model.alpha, model.w] for model in self.models]
            params = [item for sublist in params for item in sublist]
            self.optimizer = torch.optim.SGD(params, lr=lr)
            
            theta = np.exp(np.random.normal(size=(3)))*0.1
            self.shadow_model = Hawkes_mle(list(theta),T, device=device)
            
            self.shadow_params = [self.shadow_model.mu, self.shadow_model.alpha, self.shadow_model.w] 
            self.shadow_optimizer = torch.optim.SGD(self.shadow_params, lr=lr)
            
            # for evaluation
            theta = np.exp(np.random.normal(size=(3)))*0.1
            self.model_tester = Hawkes_mle(list(theta),T, device=device)
        
            self.params_tester = [self.model_tester.mu, self.model_tester.alpha, self.model_tester.w]
            self.optimizer_tester = torch.optim.SGD(self.params_tester, lr=inner_lr)
        else:
            raise NotImplementedError
            
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
                for k, model in enumerate(self.models):
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

    def update_theta(self, loss_weights):
        if self.L is None:
            print('please run compute_loss fist!')
            
        if self.method == 'mle' or self.method == 'maml':
            
            self.optimizer.zero_grad()
            
            loss = torch.sum(self.L*self.tensor(loss_weights))
            
            loss.backward()
            self.optimizer.step()
            
            return loss.data.item()
        elif self.method == 'fomaml':
            loss_accum = 0
            for user_id, tweet in enumerate(self.tweets):
#                print(user_id)
                self.optimizer.zero_grad()
                for k, (model, weight) in enumerate(zip(self.models,self.weights[user_id])):
                    params = [model.mu, model.alpha, model.w]
#                    print(wessight)
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
#                print(user_id)
                self.optimizer.zero_grad()
                for k, (model, update) in enumerate(zip(self.models,self.updates[user_id])):
                    params = [model.mu, model.alpha, model.w]
#                    print(update)
                    for param1, param2 in zip(update,params):
                        if param2.grad is None:
                            param2.grad = loss_weights[user_id,k] * param1
                        else:
                            param2.grad.data.add_(loss_weights[user_id,k] * param1)
                self.optimizer.step()
                
            return np.sum(self.L*loss_weights)
        else:
            raise NotImplementedError
            
    def show_param(self):
        for k, model in enumerate(self.models):
            print('Model:', k, "mu:",model.mu.item(),"alpha:",model.alpha.item(),"w:",model.w.item())
    
    def evaluate(self, weights):
        accum_nll = 0
        if self.method == 'mle' or self.method == 'maml':
            for model in self.models:
                model.eval()
            for i, (seq, target) in enumerate(zip(self.tweets,self.val_tweets)):
                    
                index = np.argmax(weights[i,:])
                model = self.models[index]
                #update once            
                update_mu, update_alpha, update_w = model.update_once(seq)
                
                #evaluate
                nll = model.evaluate(seq, target, update_mu, update_alpha, update_w)
    
                accum_nll += nll.data.item()
            for model in self.models:
                model.train()
                
            return accum_nll/self.N
        elif self.method == 'fomaml' or self.method == 'reptile':
            for i, (seq, target) in enumerate(zip(self.tweets,self.val_tweets)):
                self.optimizer_tester.zero_grad()
                
                index = np.argmax(weights[i,:])
                model = self.models[index]
                params = [model.mu, model.alpha, model.w]
                
                
                for param1, param2 in zip(self.params_tester,params):
                    param1.data = param2.data.clone()
                                            
                #update once
                
                loss = self.model_tester(seq)
                loss.backward()
                self.optimizer_tester.step()
                
                #evaluate
                self.model_tester.eval()
                nll = self.model_tester.evaluate(seq, target, 
                                                 self.model_tester.mu, self.model_tester.alpha, self.model_tester.w)
    
                accum_nll += nll.data.item()
                self.model_tester.train()
                
            return accum_nll/self.N
        
        
        


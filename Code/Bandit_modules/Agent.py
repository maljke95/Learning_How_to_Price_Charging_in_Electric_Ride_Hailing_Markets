# -*- coding: utf-8 -*-

from cmath import sqrt
from cmath import pi
import torch as T
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import itertools
import os
import random
from random import  uniform

from Bandit_modules.PolicyNetwork import PolicyNetwork
from Bandit_modules.ReplayBuffer import ReplayBuffer
from Bandit_modules.ActionNoise import OUActionNoise

class PolicyAgent(object):

    def __init__(self, alpha, state_dims, env, n_actions, layer1_size, layer2_size, layer3_size, batch_size, folder_name, buffer_it, 
                 min_price=0.0, max_price=10.0, add_noise=False, max_size=1300, gamma=0.99, load_weights = None) -> None:
        
        #----- Batches in the buffer -----
        
        self.buffer_it = buffer_it
        
        self.max_price = max_price
        self.min_price = min_price
        
        self.apply_random = True
        self.add_noise    = add_noise
        
        #----- Hyper parameters -----
        
        self.gamma = gamma
        self.batch_size = batch_size
        
        #----- Generate buffer -----
        
        self.memory = ReplayBuffer(max_size, state_dims, n_actions)
        
        #----- Mu network of the agent -----
        
        self.policy_mu = PolicyNetwork(alpha, state_dims, layer1_size, layer2_size, layer3_size, n_actions, folder_name+'/mu')
        
        if load_weights is not None:
            self.policy_mu.load_state_dict(T.load(load_weights))
            
        #----- Sigma network of the agent -----
        
        self.policy_sigma = PolicyNetwork(alpha, state_dims, layer1_size, layer2_size, layer3_size, n_actions, folder_name+'/sigma', softplus = True) 
        
        #----- Optimizer -----
        
        self.params = list(self.policy_mu.parameters()) + list(self.policy_sigma.parameters())
        self.optimizer = optim.Adam(self.params, lr = alpha)
        
        #----- Additional noise for exploration -----
        
        self.noise = OUActionNoise(mu=np.zeros(n_actions))
        self.list_of_all_perm = list(itertools.permutations([0,1,2,3]))
        self.ordering_idx = 0  
        
    def choose_action(self, observation, sample=False):
        
        observation = T.tensor(np.array([observation]), dtype=T.float).to(self.policy_mu.device)
        
        if not self.apply_random:
            
            self.policy_mu.eval()
            self.policy_sigma.eval()
            
            with T.no_grad():
                
                mu    = self.policy_mu.forward(observation)
                sigma = self.policy_sigma.forward(observation)
                
            self.policy_mu.train()
            self.policy_sigma.train()
            
            if self.add_noise:
                
                mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.policy_mu.device)
            
            else:
                
                mu_prime = mu 
                
            if sample:                
                action = T.normal(mean=mu_prime, std=T.clamp(sigma, min=0.0, max=0.2))
                
            else:              
                action = mu_prime
                
            #----- Clamp the action -----
            
            action = T.clamp(action, min=self.min_price, max=self.max_price)
            
            return action.cpu().detach().numpy().ravel()
        
        else:
            
            # coin = np.random.uniform(0.0, 1.0)
            # prob = 0.7
            
            # if coin >= prob:
                
            #     return self.exploration_policy5()
                
            # else:
                
            #     return self.exploration_policy4()
            
            #----- Else -----
            
            return self.exploration_policy5()      

    def choose_action_2(self, observation, sample=False):
        
        observation = T.tensor(np.array([observation]), dtype=T.float).to(self.policy_mu.device)
        
        if not self.apply_random:
            
            self.policy_mu.eval()
            self.policy_sigma.eval()
            
            with T.no_grad():
                
                mu    = self.policy_mu.forward(observation)
                sigma = self.policy_sigma.forward(observation)
                
            self.policy_mu.train()
            self.policy_sigma.train()
            
            if self.add_noise:
                
                mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.policy_mu.device)
            
            else:
                
                mu_prime = mu 
                
            if sample:                
                action = T.normal(mean=mu_prime, std=T.clamp(sigma, min=0.0, max=0.2))
                
            else:              
                action = mu_prime
                
            #----- Clamp the action -----
            
            action = T.clamp(action, min=self.min_price, max=self.max_price)
            
            return action.cpu().detach().numpy().ravel(), mu_prime.cpu().detach().numpy().ravel()
        
        else:
            
            # coin = np.random.uniform(0.0, 1.0)
            # prob = 0.7
            
            # if coin >= prob:
                
            #     return self.exploration_policy5()
                
            # else:
                
            #     return self.exploration_policy4()
            
            #----- Else -----
            
            rand_act = self.exploration_policy5()
            
            return rand_act.ravel(), rand_act.ravel()
        
    def remember(self, state, action, reward, done):
        
        if self.apply_random:
            self.memory.store_transition(state, action, reward, done, distribution=None, thr=0.70)
            
        else:
            
            coin = np.random.uniform(0.0, 1.0)
            prob = 0.5

            if coin >= prob:
                
                return self.memory.store_transition(state, action, reward, done, distribution=None, thr=0.80)#80
                
            else:
                
                return self.memory.store_transition(state, action, reward, done, distribution=None, thr=0.90)#90
            
            #----- Else -----
            
            #return self.memory.store_transition(state, action, reward, done, distribution=None, thr=0.90)
        
    def learn(self):
        
        #----- Exploration phase -----
        
        if self.memory.mem_cntr < self.buffer_it * self.batch_size:
            
            self.apply_random = True
            return
        
        #----- NN phase -----
        
        if self.apply_random:
            
            print("Learning begins...")
        
        self.apply_random = False
        
        #----- Prepare training data -----
        
        state, action, reward, done = self.memory.sample_buffer(self.batch_size)
        
        #----- Convert to tensor -----
        
        reward = T.tensor(reward, dtype=T.float).to(self.policy_mu.device)
        state  = T.tensor(state, dtype=T.float).to(self.policy_mu.device)
        action = T.tensor(action, dtype=T.float).to(self.policy_mu.device)
        
        #----- Multiple epochs in one episode -----
        
        N_ep = 30
        
        for i_ep in range(N_ep):
        
            policy_gradient = []
            log_probs = []
            
            #----- Evaluate for the given state -----
            
            mu = self.policy_mu.forward(state)
            sigma = self.policy_sigma.forward(state)
            
            for i in range(self.batch_size):
                
                p1 = -T.pow((mu[i] - action[i].reshape([1,4])), 2)/(T.clamp(2*sigma[i]**2, 1e-3))
                p2 = -T.log(sigma[i])
                
                log_probs.append(sum(p1+p2))
                
            for log_prob, r in zip(log_probs, reward):
                policy_gradient.append(-log_prob*r)         #----- I added '-' sign
            
            #----- Policy optimization -----
            
            self.policy_mu.optimizer.zero_grad()
            self.policy_sigma.optimizer.zero_grad()
            
            policy_loss = T.stack(policy_gradient).mean()
            policy_loss.backward()
            
            self.policy_mu.optimizer.step()
            self.policy_sigma.optimizer.step()
            
            self.policy_mu.scheduler.step(policy_loss)
            self.policy_sigma.scheduler.step(policy_loss)
        
    def save_models(self, iteration):

        self.policy_mu.save_checkpoint(iteration)   
        self.policy_sigma.save_checkpoint(iteration)


    def load_models(self, mu_name, sigma_name):

        self.policy_mu.load_checkpoint(mu_name)   
        self.policy_sigma.load_checkpoint(sigma_name)    
        
    def exploration_policy1(self, N_des=None, match_demand=True):
        
        act_rand = np.random.uniform(0.0, 1.0, 4)
        
        if match_demand:
            
            self.ordering_idx = 21
            
            act_rand = np.sort(act_rand)
            ordering = self.list_of_all_perm[self.ordering_idx]
            act_rand = np.array([x for y, x in sorted(zip(list(ordering), list(act_rand)))])
            
        elif N_des is not None:
            
            act_rand = np.sort(act_rand)
            ordering = np.argsort(N_des)
            act_rand = np.array([x for y, x in sorted(zip(list(ordering), list(act_rand)))])
            
        
        return act_rand            

    def exploration_policy2(self, N_des=None, match_demand=True):
        
        act_rand = np.random.uniform(0.0, 1.0, 4)
        act_rand = act_rand/np.linalg.norm(act_rand, 2)
        
        norm = random.randint(1, int(np.floor(self.max_price)))
        act_rand *= norm
        
        if match_demand:
            
            self.ordering_idx = 21
            
            act_rand = np.sort(act_rand)
            ordering = self.list_of_all_perm[self.ordering_idx]
            act_rand = np.array([x for y, x in sorted(zip(list(ordering), list(act_rand)))])
            
        elif N_des is not None:
            
            act_rand = np.sort(act_rand)
            ordering = np.argsort(N_des)
            act_rand = np.array([x for y, x in sorted(zip(list(ordering), list(act_rand)))])
            
        return act_rand            
    
    def exploration_policy3(self):
        
        act_rand = np.random.uniform(0.0, 1.0, 4)
        act_rand = act_rand/np.linalg.norm(act_rand, 2)
        
        norm = random.randint(1, int(np.floor(self.max_price)))
        act_rand *= norm
            
        return act_rand  
    
    def exploration_policy4(self):#No
        
        act_rand = np.random.uniform(0.0, self.max_price, 4)
            
        return act_rand 
    
    def exploration_policy5(self, N_des=None, match_demand=True):
        
        act_rand = np.random.uniform(0.0, self.max_price, 4)
        
        if match_demand:
            
            self.ordering_idx = 21
            
            act_rand = np.sort(act_rand)
            ordering = self.list_of_all_perm[self.ordering_idx]
            act_rand = np.array([x for y, x in sorted(zip(list(ordering), list(act_rand)))])
            
        elif N_des is not None:
            
            act_rand = np.sort(act_rand)
            ordering = np.argsort(N_des)
            act_rand = np.array([x for y, x in sorted(zip(list(ordering), list(act_rand)))])
            
        return act_rand 
    
    def exploration_policy6(self, N_des=None, match_demand=True):#No
        
        max_price_rand = random.choice([2.0, 3.0, 4.0, 5.0])
        act_rand = np.random.uniform(0.0, max_price_rand, 4)
            
        return act_rand #No
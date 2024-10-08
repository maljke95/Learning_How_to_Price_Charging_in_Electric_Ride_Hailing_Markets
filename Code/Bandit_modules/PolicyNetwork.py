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

from Bandit_modules.ReplayBuffer import ReplayBuffer
from Bandit_modules.ActionNoise import OUActionNoise

class PolicyNetwork(nn.Module):
    
    def __init__(self, alpha, state_dims, fc1_dims, fc2_dims, fc3_dims, n_actions, ckpt_file_name, softplus=False):
        
        super(PolicyNetwork, self).__init__()

        self.num_actions = n_actions
        self.state_dims = state_dims
        
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        
        self.softplus = softplus
        
        self.ckpt_file_name = ckpt_file_name

        if not os.path.isdir(self.ckpt_file_name):
            os.makedirs(self.ckpt_file_name) 
            
        self.bn0_state = nn.LayerNorm(*state_dims)

        self.fc1 = nn.Linear(*state_dims, self.fc1_dims)
        T.nn.init.xavier_uniform_(self.fc1.weight)      
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        T.nn.init.xavier_uniform_(self.fc2.weight)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        T.nn.init.xavier_uniform_(self.fc3.weight)
        self.bn3 = nn.LayerNorm(self.fc3_dims)

        self.fc4 = nn.Linear(self.fc3_dims, n_actions)
        T.nn.init.xavier_uniform_(self.fc4.weight)

        self.optimizer = optim.Adam(self.parameters(), lr = alpha)
        self.scheduler = T.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        
        self.to(self.device)

    def forward(self, state):
            
        x = self.bn0_state(state)

        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.fc4(x)

        if self.softplus:
            
            x = F.softplus(x)
            
        else:
            
            x = T.sigmoid(x)*5.0

        return x

    def save_checkpoint(self, iteration):
        
        #print('... saving checkpoint ...')
        
        if self.softplus:
            T.save(self.state_dict(), self.ckpt_file_name + '/sigma_' + str(iteration) + '.pt')
        else:
            T.save(self.state_dict(), self.ckpt_file_name + '/mu_' + str(iteration) + '.pt')
        
    def load_checkpoint(self, ckpt_file_name):
        
        #print('... loading checkpoint ...')
        self.load_state_dict(T.load(ckpt_file_name))
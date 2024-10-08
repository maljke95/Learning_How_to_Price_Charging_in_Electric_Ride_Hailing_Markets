# -*- coding: utf-8 -*-

import numpy as np

class ReplayBuffer(object):
    
    def __init__(self, max_size, state_shape, n_actions, distribution_shape=None):
        
        self.total_iteration_cntr = 0
        
        #----- Max mem size and current occupancy -----
        
        self.mem_size = max_size
        self.mem_cntr = 0
        
        self.state_memory = np.zeros((self.mem_size, *state_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)
        
        #----- Memory of desired distribution to match, important for later -----
        
        self.distribution_memory = None
        
        if distribution_shape is not None:
            
            self.distribution_memory = np.zeros((self.mem_size, *distribution_shape))
        
    def check_condition(self, state, distribution, action, reward, thr=-10.0):
        
        if reward > thr:
            
            return True
        
        else:
            
            return False
        
    def store_transition(self, state, action, reward, done=1, distribution=None, thr=-10.0):
        
        self.total_iteration_cntr += 1
        
        write_in_mem = self.check_condition(state, distribution, action, reward, thr)
        
        if write_in_mem:
            
            if self.mem_cntr < self.mem_size:
                
                index = self.mem_cntr
                
            else:
                
                index = self.mem_cntr % self.mem_size
                
            self.mem_cntr += 1
            
            self.state_memory[index]    = state
            self.action_memory[index]   = action
            self.reward_memory[index]   = reward  
            self.terminal_memory[index] = 1 - done
            
            if self.distribution_memory is not None:
                
                self.distribution_memory[index] = distribution
                
    def sample_buffer(self, batch_size):
        
        max_mem = min(self.mem_cntr, self.mem_size)
        
        batch = np.random.choice(max_mem, batch_size, replace=False)
                
        states = self.state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch] 
        terminal = self.terminal_memory[batch]
        
        if self.distribution_memory is not None:
            
            distributions = self.distribution_memory[batch]
            return states, actions, rewards, terminal, distributions
        
        else:
            
            return states, actions, rewards, terminal
            
            
            
                
            
            
        
        
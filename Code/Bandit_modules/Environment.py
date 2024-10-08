# -*- coding: utf-8 -*-

from gym import spaces, Env
import numpy as np
import os
import copy

from Charging_balancing_modules.utils_s import *
from Charging_balancing_modules.Government_s import *
from Charging_balancing_modules.Company_s import *
from Charging_balancing_modules.Optimizer_s import *
from Charging_balancing_modules.GameCoord_s import *
from Charging_balancing_modules.CompanyLoss_s import *
from Charging_balancing_modules.Game_s import *

class RHEnvironment(Env):
    
    def __init__(self, N, M, max_price, min_price, episode_length, warm_start_filename=None, warm_start=True):
        
        super(RHEnvironment, self).__init__()
        
        self.step_cntr = 0
        
        #----- Init parameters -----
        
        self.episode_length = episode_length
        self.N = N
        self.M = M
        self.max_price = max_price
        self.min_price = min_price
        
        #----- To mathc the gym environment standard we define these -----
        
        self.observation_shape = (N*M,)
        self.action_shape = (M,)
        
        self.observation_space = spaces.Box(low   = np.zeros(self.observation_shape), 
                                            high  = np.ones(self.observation_shape),
                                            dtype = np.float32)
        
        self.action_space = spaces.Box(low   = min_price*np.ones(self.action_shape), 
                                       high  = max_price*np.ones(self.action_shape),
                                       dtype = np.float32)
        
        #----- Current values -----
        
        self.current_state = None 
        self.obs = None 
        self.reward = None
        
        #----- Using already prepared data -----
        
        if warm_start:
            
            self.warm_start(warm_start_filename)
            
    def warm_start(self, warm_start_filename):
        
        self.artificial_state = False
        self.warm_start_filename = warm_start_filename
        
        self.g = None
        
        self.Ag = None
        self.bg = None
        self.p_occ_belief = None
        self.avg_earning_belief = None
        self.Q = None
        self.N_max = None
        self.N_des = None
        self.Map = None
        
        self.RHVs = None
        
        self.gammas = None
        self. N_companies = None
        
        if not warm_start_filename is None:
            
            self.warm_reset()
            self.next_observation() 
            
    def warm_reset(self):
        
        current_folder = os.getcwd()+'/Results'
        name = current_folder + "/" + self.warm_start_filename        
            
        if self.g is None:
            self.g = Game() 
            
        #----- Load info about the game -----
        
        Ag, bg, p_occ_belief, avg_earning_belief, Q, N_max, N_des, Map, RHVs, gammas, N_companies, fixed_price = self.g.load_info_about_game(name)
        
        self.Ag = Ag
        self.bg = bg
        self.p_occ_belief = p_occ_belief
        self.avg_earning_belief = avg_earning_belief
        self.Q = Q
        self.N_max = N_max
        self.N_des = N_des
        self.Map = Map
        
        self.RHVs = RHVs
        
        self.list_of_cbl_orig = []
        self.list_of_dbl_orig = []
        self.list_of_mxr_orig = []
        
        for idx, veh in enumerate(self.RHVs):
                
            self.list_of_cbl_orig.append(veh.current_battery_level) 
            self.list_of_dbl_orig.append(veh.desired_battery_level) 
            self.list_of_mxr_orig.append(veh.max_range)               
        
        self.gammas = gammas
        self. N_companies = N_companies   
        
        self.list_of_company_IDs = []
        
        for veh in self.RHVs:
            self.list_of_company_IDs.append(veh.companyID)
        
        #----- Desired vehicle distribution -----
        
        self.desired_distribution = Map.N_des_coeff
        
        #----- Reset the game -----
        
        self.g.reset_game()
        
    def step(self, action):
        
        self.step_cntr += 1

        if self.step_cntr % self.episode_length == 0:
            done = True 
        else:
            done = False
        
        #----- The reward for the actual action -----
        
        #self.g.reset_game()
        reward, remaining_info = self.get_reward(action)
        
        new_state = self.get_new_state()
        info = None
        
        return new_state, reward, done, info, remaining_info

    def step_2(self, action, mean_act=None):
        
        self.step_cntr += 1

        if self.step_cntr % self.episode_length == 0:
            done = True 
        else:
            done = False
        
        #----- Check the reward if no sampling is done -----
        
        if mean_act is not None:
            
            #self.g.reset_game()
            reward_mean, remaining_info_mean = self.get_reward(mean_act)
        
        #----- The reward for the actual action -----
        
        #self.g.reset_game()
        reward, remaining_info = self.get_reward(action)
        
        new_state = self.get_new_state()
        info = None
        
        return new_state, reward, done, info, remaining_info, reward_mean, remaining_info_mean
    
    def reset(self, artificial_state=True):
        
        self.artificial_state = artificial_state       

        return self.obs
    
    def get_reward(self, action, N_ref=None):
        
        fixed_price = action

        name = None
        
        self.g.reset_game()
        
        N_ref = self.desired_distribution
        if N_ref is not None:
        
            N_veh = len(self.RHVs)
            N_des = np.array(len(N_ref)*[0.0])
        
            for i in range(len(N_ref)-1):
            
                N_des[i] = np.floor(N_veh/np.sum(N_ref)*N_ref[i])
            
            N_des[-1] = N_veh - np.sum(N_des)
            self.N_des = N_des
            
        else:
            
            self.N_des = self.desired_distribution
        
        _ = self.g.Run_game_from_stored_data(self.Ag, self.bg, self.p_occ_belief, self.avg_earning_belief, self.Q, 
                                                                         self.N_max, self.N_des, self.Map, self.RHVs, self.gammas, self.N_companies, 
                                                                         fixed_price, plot_graphs=False, list_of_games=[1], list_of_games_to_play=[1], 
                                                                         folder_name=name, robust_coeff=0.0, scale_param=1.0)
        
        #----- Distributions -----
        
        desired_distribution  = self.N_des
        desired_distribution  /= np.sum(desired_distribution)
        
        achieved_distribution = self.g.list_of_GameCoord[0].sigma_x
        achieved_distribution /= np.sum(achieved_distribution)
        
        reward =  1.0 - np.linalg.norm(desired_distribution-achieved_distribution, 2)/(2.0**0.5)
        #reward =  1.0 - np.linalg.norm(desired_distribution-achieved_distribution, 1)/(2.0)
        
        #----- Remaining info -----
        
        remaining_info = {}
        remaining_info['desired_distribution'] = desired_distribution
        remaining_info['achieved_distribution'] = achieved_distribution
        
        return reward, remaining_info
    
    def get_new_state(self):
        
        if self.artificial_state:
            
            company_ID_perm = np.random.permutation(self.list_of_company_IDs)
            for idx, veh in enumerate(self.RHVs):
                
                self.RHVs[idx].companyID = company_ID_perm[idx]
                self.RHVs[idx].current_battery_level = self.list_of_cbl_orig[idx] + np.random.uniform(-10.0, 10.0)
                self.RHVs[idx].desired_battery_level = self.list_of_dbl_orig[idx] + np.random.uniform(-10.0, 10.0)
                self.RHVs[idx].max_range             = self.list_of_mxr_orig[idx] + np.random.uniform(-10.0, 10.0) 
        
        #----- Generate next state -----
        
        self.next_observation()
        new_state = np.copy(self.obs) 
        
        return new_state
    
    def next_observation(self):
        
        game_id = 1
        
        gc1 = GameCoordinator(self.Map, game_id, self.N_des)
        G1 = Government()
        G1.generate_government(np.copy(self.Ag), np.copy(self.bg))
        gc1.set_gov(G1) 
        
        list_of_Di = None
        list_of_fi = None
        
        for i in range(self.N_companies):
                
            c1 = Company(self.Map, i)
            c1.convert_to_vehicles_from_sim(self.RHVs)
            c1.generate_feasibility_set()
            c1.prepare_cumulative_state(np.copy(self.p_occ_belief[i]), np.copy(self.avg_earning_belief[i]))
            Ai1, Bi1, ci1, Di1, fi1, Ni1 = c1.generate_matrices_for_loss(self.N_max, self.N_des, self.Q)  
            
            #Di1 /= np.sum(np.sum(Di1))
            #fi1 /= np.sum(fi1)
            
            if list_of_Di is None:
                list_of_Di = np.squeeze(np.copy(np.diag(Di1)))
            else:
                list_of_Di = np.append(list_of_Di, np.squeeze(np.copy(np.diag(Di1))))
                
            if list_of_fi is None:             
                list_of_fi = np.squeeze(np.copy(fi1))          
            else:
                list_of_fi = np.append(list_of_fi, np.squeeze(np.copy(fi1)))
                
        
        observation = np.copy(list_of_Di)
        observation = np.append(observation, list_of_fi)
        
        self.obs = observation
        

        
    
        
                
        
        
        
        
        
        
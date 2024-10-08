# -*- coding: utf-8 -*-

from cmath import log
import numpy as np
import os
import sys
import torch as T
import gym
from collections import defaultdict
import pickle
import time
from datetime import datetime 
import matplotlib.pyplot as plt
import tikzplotlib

from Bandit_modules.Environment import RHEnvironment as Environment
from Bandit_modules.PolicyNetwork import PolicyNetwork
from Bandit_modules.Agent import PolicyAgent

def main(sample_f=False, pretrain=False, pre_buffers=1, batch_size=32, save_plot=True):
    
    save_plot =True
    np.random.seed(0)
    
    N = 3
    M = 4
    min_price = 0.0
    max_price = 5.0
    episode_length = 8
    
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    
    #----- Generate env -----
    
    warm_start_filename = "/03_23_2022_21_08_11" 
    env = Environment(N, M, max_price, min_price, episode_length, warm_start_filename, warm_start=True)
    
    #----- Setup agent -----
    
    alpha = 0.001
    
    state_dims = [24]
    
    
    layer1_size = 256
    layer2_size = 64
    layer3_size = 16
    n_actions = 4
    n_episodes = 1000

    buffer_it = pre_buffers    
    
    current_folder = os.getcwd()
    load_folder_name = current_folder +"/Results" + warm_start_filename

    #----- Load weights pretrained -----
    
    pretrained_mu_filename = "/mu/pretrained/mu_30.pt" 
    load_weights = load_folder_name + pretrained_mu_filename
    
    #----- Setup algorithm -----
    
    mov_average = False
    pretrained_weights = False
    add_noise = False

    #----- Result statistic files folder -----
    
    if not os.path.exists(current_folder + "/Bandit_result/" + date_time +"/UniqueDistri"):
        os.makedirs(current_folder + "/Bandit_result/" + date_time +"/UniqueDistri")
    folder_name_result = current_folder + "/Bandit_result/" + date_time +"/UniqueDistri"
    
    if pretrained_weights:
        agent = PolicyAgent(alpha, state_dims, env, n_actions, layer1_size,
                    layer2_size, layer3_size, batch_size, folder_name_result, buffer_it, min_price, max_price, add_noise=add_noise, max_size=1300, gamma=0.99, load_weights=load_weights) 
    
    else:
        agent = PolicyAgent(alpha, state_dims, env, n_actions, layer1_size,
                    layer2_size, layer3_size, batch_size, folder_name_result, buffer_it, min_price, max_price, add_noise=add_noise, max_size=1300, gamma=0.99)
        
    #----- Learning procedure -----
    
    score_history = []
    average_score = []
    achieved_dist_evolution = []
    desired_dist_evolution = []
    
    achieved_dist_mean_act_evolution = []
    score_history_mean = []
    
    score_buffer = []  
    score_sum = 0.0
    
    mean_act_activate = False
    
    for i in range(n_episodes):
        
        done = False
        
        #----- Get observation -----
        
        obs = env.reset()
        
        #----- Choose action 1 -----
        
        if mean_act_activate:
        
            act, mean_act = agent.choose_action_2(np.array(obs),sample=sample_f)
            new_state, reward, done, info, remaining_info, reward_mean, remaining_info_mean = env.step_2(act, mean_act)
        
        else:
            
           act = agent.choose_action(np.array(obs),sample=sample_f)
           new_state, reward, done, info, remaining_info = env.step(act)
        
        #----- Interact with the environment -----

        desired_distribution  = remaining_info['desired_distribution']
        achieved_distribution = remaining_info['achieved_distribution']
        
        if mean_act_activate:
            
            achieved_distribution_mean_act = remaining_info_mean['achieved_distribution']
            score_mean_act = reward_mean
        
        score = reward
        
        #----- Log the statistics -----
        
        achieved_dist_evolution.append(achieved_distribution)
        desired_dist_evolution.append(desired_distribution)
        score_history.append(score)
        
        average_score.append(np.mean(score_history[-100:]))
        
        if mean_act_activate:
            
            achieved_dist_mean_act_evolution.append(achieved_distribution_mean_act)
            score_history_mean.append(score_mean_act)
        
        #----- Add to buffer -----
        
        intelligent_buffer = True
        
        if intelligent_buffer:
        
            if mov_average:
                
                if i == 0:      
                    agent.remember(np.array(obs), act, reward, int(done))
                    
                elif reward > score_sum/i:
                    agent.remember(np.array(obs), act, reward, int(done))
                    
                score_sum += reward
                    
            else:
                
                if i < buffer_it * batch_size:
                    score_buffer.append(reward)
                    agent.remember(np.array(obs), act, reward, int(done)) 
                    
                else:
                    
                    print('Buffer mean is: '+str(np.mean(score_buffer))+' and reward is '+str(reward))
                    
                    if reward > np.mean(score_buffer):
                        score_buffer.append(reward)
                        agent.remember(np.array(obs), act, reward, int(done)) 
                    
        else:
                
            agent.remember(np.array(obs), act, reward, int(done)) 
                
        #----- Train agent -----
        
        agent.learn()
        
        print('Ep: ',i, '| s:%.3f' %reward, '|Act: %.2f' %act[0], ' %.2f' %act[1], ' %.2f' %act[2], ' %.2f' %act[3])
        
        if i % 100 == 99:
            
            np.save(folder_name_result + '/average_score.npy', np.array(average_score))
            np.save(folder_name_result + '/score_history.npy', np.array(score_history))
            np.save(folder_name_result + '/achieved_distribution.npy', np.array(achieved_dist_evolution))
            np.save(folder_name_result + '/desired_distribution.npy', np.array(desired_dist_evolution))
            agent.save_models(i)
            
            if mean_act_activate:
                
                #----- Mean results -----
            
                np.save(folder_name_result + '/score_history_mean.npy', np.array(score_history_mean))
                np.save(folder_name_result + '/achieved_distribution_mean.npy', np.array(achieved_dist_mean_act_evolution))
    
    if save_plot:
        
        k = np.arange(len(average_score))
        
        fig1, ax1 = plt.subplots(dpi=180)
        
        ax1.scatter(k, score_history)    
        ax1.grid('on')
        ax1.legend()
        ax1.set_xlabel(r'$k$')
        ax1.set_ylabel(r'$R$')    
        
        fig1.savefig(folder_name_result + "/score_history.jpg", dpi=180)
        tikzplotlib.save(folder_name_result + "/score_history.tex")
        
        fig2, ax2 = plt.subplots(dpi=180)
        
        ax2.plot(k, average_score)    
        ax2.grid('on')
        ax2.legend()
        ax2.set_xlabel(r'$k$')
        ax2.set_ylabel(r'$R$') 
        
        fig2.savefig(folder_name_result + "/average_score.jpg", dpi=180)
        tikzplotlib.save(folder_name_result + "/average_score.tex")
        
        plt.show()
        plt.close()        
    
    return np.array(average_score), np.array(score_history), np.array(achieved_dist_evolution), np.array(desired_dist_evolution)
    
def plot_performance(average_score, score_history, achieved_dist_evolution, desired_dist_evolution):
    
    k = np.arange(len(average_score))
    
    fig1, ax1 = plt.subplots(dpi=180)
    
    ax1.scatter(k, score_history)    
    ax1.grid('on')
    ax1.legend()
    ax1.set_xlabel(r'$k$')
    ax1.set_ylabel(r'$R$')    
    
    fig2, ax2 = plt.subplots(dpi=180)
    
    ax2.plot(k, average_score)    
    ax2.grid('on')
    ax2.legend()
    ax2.set_xlabel(r'$k$')
    ax2.set_ylabel(r'$R$') 
    
    plt.show()
    
def analyse_results():
    
    date_time = '03_28_2023_21_54_09'
    
    current_folder = os.getcwd()
    folder_name_result = current_folder + "/Bandit_result/" + date_time +"/UniqueDistri"
    
    average_score           = np.load(folder_name_result + '/average_score.npy')
    score_history           = np.load(folder_name_result + '/score_history.npy')
    achieved_dist_evolution = np.load(folder_name_result + '/achieved_distribution.npy')
    desired_dist_evolution  = np.load(folder_name_result + '/desired_distribution.npy')
    
    list_of_colors = ['tab:red', 'tab:orange', 'tab:gray', 'tab:cyan']
    
    k = np.arange(len(average_score))
    
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(nrows=2, ncols=2, dpi=180)
    
    ax  = [ax1,ax2,ax3,ax4]

    N_exp = 250
    y_min1 = 0.0
    y_max1 = 0.6
        
    for i in range(4):
        
        ax_curr  = ax[i]
        
        color_i = list_of_colors[i]
        achieved_distr = achieved_dist_evolution[:,i]
        desired_dist   = desired_dist_evolution [:,i]
        
        ax_curr.scatter(k, achieved_distr[:len(k)], color='k', marker='x', s=1, label = 'H '+str(i+1), alpha=0.3)
        ax_curr.plot(k, desired_dist  [:len(k)], '--', linewidth=3, color=color_i)
    
        ax_curr.vlines(N_exp, ymin=y_min1, ymax=y_max1, linestyles ='dashed')
    
        ax_curr.grid('on')
        ax_curr.legend()
        
        if i in [2,3]:
            ax_curr.set_xlabel(r'$k$')
        
        if i in [0,2]:
            ax_curr.set_ylabel(r'$R$')
            
        if i in [0,1]:
            ax_curr.set_xticklabels([])
            
        if i in [1,3]:
            ax_curr.set_yticklabels([])
            
        ax_curr.set_xlim(0,len(k))
        ax_curr.set_ylim(0.0, 0.6) 
        ax_curr.legend(loc='upper right')
    

    
    fig.savefig(folder_name_result + "/distribution.jpg", dpi=180)
    tikzplotlib.save(folder_name_result + "/distribution.tex")   
    
    fig22, ax21 = plt.subplots(dpi=180)
    
    k_random = np.arange(250)
    reward_random = score_history[:250]
    
    ax21.scatter(k_random,reward_random, color='k', marker='x', s=3,label='Exploration')
    ax21.vlines (N_exp, ymin=y_min1, ymax=1.0, linestyles ='dashed')
    ax21.plot(k, average_score, color='b', label='Moving average')
    
    ax21.set_xlim(0,len(k))
    ax21.set_ylim(0.6, 1.0) 
    ax21.grid('on')
    ax21.set_xlabel(r'$k$')
    ax21.set_ylabel(r'$R$')
    ax21.legend(loc='lower right')

    fig22.savefig(folder_name_result + "/reward.jpg", dpi=180)
    tikzplotlib.save(folder_name_result + "/reward.tex")
    
    plt.show()
    
def test_once():

    np.random.seed(0)
    
    N = 3
    M = 4
    min_price = 0.0
    max_price = 5.0
    episode_length = 8
    
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    
    #----- Generate env -----
    
    warm_start_filename = "/03_23_2022_21_08_11" 
    env = Environment(N, M, max_price, min_price, episode_length, warm_start_filename, warm_start=True)
    
    #----- Setup agent -----
    
    alpha = 0.001
    
    state_dims = [24]
    
    
    layer1_size = 256
    layer2_size = 64
    layer3_size = 16
    n_actions = 4
    n_episodes = 1000

    buffer_it = 5    
    
    current_folder = os.getcwd()
    load_folder_name = current_folder +"/Results" + warm_start_filename

    #----- Load weights pretrained -----
    
    pretrained_mu_filename = "/mu/pretrained/mu_30.pt" 
    load_weights = load_folder_name + pretrained_mu_filename
    
    #----- Setup algorithm -----
    
    mov_average = False
    pretrained_weights = False
    add_noise = False

    #----- Result statistic files folder -----
    
    batch_size = 32
    
    if not os.path.exists(current_folder + "/Bandit_result/" + date_time +"/UniqueDistri"):
        os.makedirs(current_folder + "/Bandit_result/" + date_time +"/UniqueDistri")
    folder_name_result = current_folder + "/Bandit_result/" + date_time +"/UniqueDistri"
    
    if pretrained_weights:
        agent = PolicyAgent(alpha, state_dims, env, n_actions, layer1_size,
                    layer2_size, layer3_size, batch_size, folder_name_result, buffer_it, min_price, max_price, add_noise=add_noise, max_size=1300, gamma=0.99, load_weights=load_weights) 
    
    else:
        agent = PolicyAgent(alpha, state_dims, env, n_actions, layer1_size,
                    layer2_size, layer3_size, batch_size, folder_name_result, buffer_it, min_price, max_price, add_noise=add_noise, max_size=1300, gamma=0.99)
    
    mu_name    = os.getcwd() + "/Bandit_result/" + "03_28_2023_21_54_09" +"/UniqueDistri" + "/mu"    + "/mu_999.pt"
    sigma_name = os.getcwd() + "/Bandit_result/" + "03_28_2023_21_54_09" +"/UniqueDistri" + "/sigma" + "/sigma_999.pt"
    
    agent.load_models(mu_name, sigma_name)
    
    
    agent.apply_random = False
    
    obs = env.reset()
    act = agent.choose_action(np.array(obs),sample=False)
    new_state, reward, done, info, remaining_info = env.step(act)
    achieved_distribution = remaining_info['achieved_distribution']
    
    print(reward)
    print(act)
    print(achieved_distribution)
    

if __name__ == '__main__': 

    #average_score, score_history, achieved_dist_evolution, desired_dist_evolution = main(sample_f=True, pretrain=False, pre_buffers=5, batch_size=32, save_plot=True)        
    
    #analyse_results()
    
    test_once()

    
    
    
    
    
    
# %%
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from kernel_MDP import *

import gym

from tqdm.notebook import tqdm

import copy
import math

import pickle as pkl

num_agents = 1
dataset_len = 4000
lam = 1

min_position = -1.2
max_position = 0.6
max_speed = 0.07
# goal_position = 0.5
goal_position = 0.45
goal_velocity = 0.0
goal_rewards = 0.0

min_action = -1.0
max_action = 1.0

power = 0.0015

force = 0.001
gravity = 0.0025
# %%
action_grid_dim = 10
# actions = np.array([0,1,2]).reshape(-1,1)
actions = np.linspace(start=-1,stop=1,num=action_grid_dim).reshape(-1,1)
n_action = len(actions)

# np.random.seed(0)
pos = np.random.uniform(low=min_position,high=max_position,size=(dataset_len,1))
vel = np.random.uniform(low=-max_speed,high=max_speed,size=(dataset_len,1))
# u = np.random.choice(3,size=(dataset_len,1))
u = np.random.uniform(low=-1,high=1,size=(dataset_len,1))
# single_act = 2
# u = np.ones_like(pos)*actions[single_act]

X = np.hstack((pos,vel,u))
X_s = np.hstack((pos,vel))

Y = np.zeros_like(X_s)

for k in range(dataset_len):
    force = min(max(u[k], min_action), max_action)
    new_vel = vel[k] + force * power - 0.0025 * math.cos(3 * pos[k])
    if new_vel > max_speed:
        new_vel = max_speed
    if new_vel < -max_speed:
        new_vel = -max_speed
    new_pos = pos[k] + new_vel
    if new_pos > max_position:
        new_pos = max_position
    if new_pos < min_position:
        new_pos = min_position
    if new_pos == min_position and new_vel < 0:
        new_vel = 0
    Y[k,0] = new_pos
    Y[k,1] = new_vel

X_scaler = StandardScaler()
X_scaler.fit(X)

X_s_scaler = StandardScaler()
X_s_scaler.fit(X_s)

sigma_div = 10
lam = (1/len(X))*100

scaling = True

if scaling:
    X_fit = X_scaler.transform(X)
    X_s_fit = X_s_scaler.transform(X_s)
else:
    X_fit = X
    X_s_fit = X_s

sigma_k, gamma_k = kw_mean_dist(X_fit,sigma_div)

K = rbf_kernel(X_fit,gamma=gamma_k)

W = np.linalg.inv(K + lam*len(X)*np.eye(K.shape[0]))

if scaling:
    Y_fit = X_s_scaler.transform(Y)
else:
    Y_fit = Y

Y_rep = np.repeat(Y,len(actions),axis=0)
A_rep = np.tile(actions,(np.shape(X_s)[0],1))

if scaling:
    Y_test = X_scaler.transform(np.hstack((Y_rep,A_rep)))
else:
    Y_test = np.hstack((Y_rep,A_rep))

alphas = (W@rbf_kernel(X_fit,Y_test,gamma=gamma_k)).T + 0.0000000001
alphas = alphas/np.abs(alphas).sum(axis=1,keepdims=True)
alphas = alphas.reshape((dataset_len,n_action,dataset_len))

costs_Y = np.zeros((dataset_len,n_action))

for s,state in enumerate(Y):
    for a,act in enumerate(actions):
        force = min(max(act, min_action), max_action)
        new_vel = state[1] + force * power - 0.0025 * math.cos(3 * state[0])
        if new_vel > max_speed:
            new_vel = max_speed
        if new_vel < -max_speed:
            new_vel = -max_speed
        new_pos = state[0] + new_vel
        if new_pos > max_position:
            new_pos = max_position
        if new_pos < min_position:
            new_pos = min_position
        if new_pos == min_position and new_vel < 0:
            new_vel = 0
        
        if new_pos >= goal_position and new_vel >= goal_velocity:
            costs_Y[s][a] = -100.0
        else:
            costs_Y[s][a] = math.pow(act, 2) * 0.1

V_init = np.random.uniform((dataset_len,1))*10
V,policy,val_iter,deltas,deltas_pol = CME_VI(Y,costs_Y,alphas,gamma=0.999,theta=0.01,max_val_iter=10000,err=np.infty)
# V,policy,val_iter,deltas,deltas_pol = CME_VI(Y,costs_Y,alphas,gamma=0.999,theta=0.01,max_val_iter=2000,err=np.infty)

with open("mountaincar_continuous_CME_output.pkl", "wb") as out_file:
    pkl.dump([X_fit,X_scaler,actions,W,gamma_k,V], out_file)

import numpy as np
from scipy.spatial.distance import pdist, cdist
from tqdm.notebook import tqdm
import copy
from sklearn.neighbors import NearestNeighbors

import cupy


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

def calc_alphas(states,actions,D,W,sigmalist):
    n_state = states.shape[0]
    n_action = len(actions)
    nx = D.shape[0]
    alphas = np.zeros((n_state,n_action,nx))
    for k, state in tqdm(enumerate(states)):
        dists = cdist(D,np.hstack((np.tile(state,[n_action,1]),actions[:,np.newaxis])),metric='sqeuclidean')
        alphas[k] = (W@np.exp(-dists/(2*sigmalist[0]**2))).T + 0.0000000001
        alphas[k] = alphas[k]/np.abs(alphas[k]).sum(axis=1,keepdims=True)
    return alphas

def calc_rewards(states,actions):
    n_state = states.shape[0]
    n_action = len(actions)
    rewards = np.zeros((n_state,n_action))
    for k, state in tqdm(enumerate(states)):
        for i,action in enumerate(actions):
            th = state[0]
            thdot = state[1]
            rewards[k][i] = -(angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (action**2)) 
    return rewards

def calc_rewards_Gaussian(states,actions,rewards_all):
    n_state = states.shape[0]
    n_action = len(actions)
    rewards = np.zeros((n_state,n_action))
    for k, state in tqdm(enumerate(states)):
        for i,action in enumerate(actions):
            x = int(state[0])
            y = int(state[1])
            rewards[k][i] = rewards_all[x][y] 
    return rewards







## re-writing CME_VI to run on the GPU
def CME_VI(states,rewards,alphas,gamma=0.99,theta=0.001,max_val_iter=10000,err=np.infty,V_init=None):
    n_state = states.shape[0]
    if not V_init:
        V = np.zeros((n_state,1))

    policy = np.zeros(n_state)

    with cupy.cuda.Device(0):
        max_val_iter_d = max_val_iter
        theta_d = theta
        deltas_d = []
        val_iter_d = 0
        V_d = cupy.asarray(V)
        policy_d = cupy.asarray(policy)
        alphas_d = cupy.asarray(alphas)
        rewards_d = cupy.asarray(rewards)

        while val_iter_d < max_val_iter_d and err > theta_d:
            if not val_iter_d%100:
                print('CME_VI iteration ', val_iter_d)
                print(err)
            V_temp = cupy.zeros_like(V_d)
            buffer = rewards_d + gamma*cupy.squeeze(alphas_d@V_d)
            V_temp = cupy.min(buffer,axis=1)
            policy_d = cupy.argmin(buffer,axis=1)
            err = cupy.max(cupy.abs(V_d-V_temp))
            deltas_d.append(err)
            V_d = V_temp
            val_iter_d = val_iter_d +1

        V = cupy.asnumpy(V_d)
        policy = cupy.asnumpy(policy)
        val_iter = val_iter_d
        deltas = cupy.asnumpy(cupy.array(deltas_d))


    return V,policy,val_iter,deltas,deltas_pol

    

def Gaussian_MDP(rew_sigma,grid_dim=50,success_prob=0.8,):

    n_state = grid_dim**2
    n_action = 4

    t_matrix = np.zeros((n_state,n_action,n_state))
    # success_prob = 0.8
    for i in range(n_state):
        # left edge
        # if (i % grid_dim) == 0:
        #     #north west corner
        if i == 0:
            #north
            #t_matrix[i][0][i-grid_dim] = 0
            t_matrix[i][0][i+grid_dim] = 1/2
            #t_matrix[i][0][i-1] = 0
            t_matrix[i][0][i+1] = 1/2
            #east
            #t_matrix[i][1][i-grid_dim] = 0
            t_matrix[i][1][i+grid_dim] = (1-success_prob)
            #t_matrix[i][1][i-1] = 0
            t_matrix[i][1][i+1] = success_prob
            #south
            #t_matrix[i][2][i-grid_dim] = 0
            t_matrix[i][2][i+grid_dim] = success_prob
            #t_matrix[i][2][i-1] = 0
            t_matrix[i][2][i+1] = (1-success_prob)
            #west
            #t_matrix[i][3][i-grid_dim] = 0
            t_matrix[i][3][i+grid_dim] = 1/2
            #t_matrix[i][3][i-1] = 0
            t_matrix[i][3][i+1] = 1/2
        # north east corner
        elif i == (grid_dim-1):
            #north
            #t_matrix[i][0][i-grid_dim] = 0
            t_matrix[i][0][i+grid_dim] = 1/2
            t_matrix[i][0][i-1] = 1/2
            #t_matrix[i][0][i+1] = 0
            #east
            #t_matrix[i][1][i-grid_dim] = 0
            t_matrix[i][1][i+grid_dim] = 1/2 
            t_matrix[i][1][i-1] = 1/2
            #t_matrix[i][1][i+1] = 0
            #south
            #t_matrix[i][2][i-grid_dim] = 0
            t_matrix[i][2][i+grid_dim] = success_prob
            t_matrix[i][2][i-1] = (1-success_prob)
            #t_matrix[i][2][i+1] = 0
            #west
            #t_matrix[i][3][i-grid_dim] = 0
            t_matrix[i][3][i+grid_dim] = (1-success_prob)
            t_matrix[i][3][i-1] = success_prob
            #t_matrix[i][3][i+1] = 0
        # south west corner
        elif i == grid_dim * (grid_dim-1):
            #north
            t_matrix[i][0][i-grid_dim] = success_prob
            #t_matrix[i][0][i+grid_dim] = 0
            #t_matrix[i][0][i-1] = 0
            t_matrix[i][0][i+1] = 1-success_prob
            #east
            t_matrix[i][1][i-grid_dim] = 1-success_prob
            #t_matrix[i][1][i+grid_dim] = 0
            #t_matrix[i][1][i-1] = 0
            t_matrix[i][1][i+1] = success_prob
            #south
            t_matrix[i][2][i-grid_dim] = 1/2
            #t_matrix[i][2][i+grid_dim] = 0
            #t_matrix[i][2][i-1] = 0
            t_matrix[i][2][i+1] = 1/2
            #west
            t_matrix[i][3][i-grid_dim] = 1/2
            #t_matrix[i][3][i+grid_dim] = 0
            #t_matrix[i][3][i-1] = 0
            t_matrix[i][3][i+1] = 1/2
        # south east corner
        elif i == grid_dim * grid_dim - 1:
            #north
            t_matrix[i][0][i-grid_dim] = success_prob
            #t_matrix[i][0][i+grid_dim] = 0
            t_matrix[i][0][i-1] = 1-success_prob
            #t_matrix[i][0][i+1] = 0
            #east
            t_matrix[i][1][i-grid_dim] = 1/2
            #t_matrix[i][1][i+grid_dim] = 0
            t_matrix[i][1][i-1] = 1/2
            #t_matrix[i][1][i+1] = 0
            #south
            t_matrix[i][2][i-grid_dim] = 1/2
            #t_matrix[i][2][i+grid_dim] = 0
            t_matrix[i][2][i-1] = 1/2
            #t_matrix[i][2][i+1] = 0
            #west
            t_matrix[i][3][i-grid_dim] = (1-success_prob)
            #t_matrix[i][3][i+grid_dim] = 0
            t_matrix[i][3][i-1] = success_prob
            #t_matrix[i][3][i+1] = 0
        # north border not in corners
        elif i < grid_dim:
            #north 
            #t_matrix[i][0][i-grid_dim] = 0
            t_matrix[i][0][i+grid_dim] = 1/3
            t_matrix[i][0][i-1] = 1/3
            t_matrix[i][0][i+1] = 1/3
            #east
            #t_matrix[i][1][i-grid_dim] = 0
            t_matrix[i][1][i+grid_dim] = (1-success_prob)/2
            t_matrix[i][1][i-1] = (1-success_prob)/2
            t_matrix[i][1][i+1] = success_prob
            #south
            #t_matrix[i][2][i-grid_dim] = 0
            t_matrix[i][2][i+grid_dim] = success_prob
            t_matrix[i][2][i-1] = (1-success_prob)/2
            t_matrix[i][2][i+1] = (1-success_prob)/2
            #west
            #t_matrix[i][3][i-grid_dim] = 0
            t_matrix[i][3][i+grid_dim] = (1-success_prob)/2
            t_matrix[i][3][i-1] = success_prob
            t_matrix[i][3][i+1] = (1-success_prob)/2
        # west border not in corners
        elif i%grid_dim==0:
            #north 
            t_matrix[i][0][i-grid_dim] = success_prob
            t_matrix[i][0][i+grid_dim] = (1-success_prob)/2
            #t_matrix[i][0][i-1] = 0
            t_matrix[i][0][i+1] = (1-success_prob)/2
            #east
            t_matrix[i][1][i-grid_dim] = (1-success_prob)/2
            t_matrix[i][1][i+grid_dim] = (1-success_prob)/2
            #t_matrix[i][1][i-1] = 0
            t_matrix[i][1][i+1] = success_prob
            #south
            t_matrix[i][2][i-grid_dim] = (1-success_prob)/2
            t_matrix[i][2][i+grid_dim] = success_prob
            #t_matrix[i][2][i-1] = 0
            t_matrix[i][2][i+1] = (1-success_prob)/2
            #west
            t_matrix[i][3][i-grid_dim] = 1/3
            t_matrix[i][3][i+grid_dim] = 1/3
            #t_matrix[i][3][i-1] = 0
            t_matrix[i][3][i+1] = 1/3
        # east border not in corners
        elif (i+1)%grid_dim==0:
            #north 
            t_matrix[i][0][i-grid_dim] = success_prob
            t_matrix[i][0][i+grid_dim] = (1-success_prob)/2
            t_matrix[i][0][i-1] = (1-success_prob)/2
            #t_matrix[i][0][i+1] = 0
            #east
            t_matrix[i][1][i-grid_dim] = 1/3
            t_matrix[i][1][i+grid_dim] = 1/3
            t_matrix[i][1][i-1] = 1/3
            #t_matrix[i][1][i+1] = 0
            #south
            t_matrix[i][2][i-grid_dim] = (1-success_prob)/2
            t_matrix[i][2][i+grid_dim] = success_prob
            t_matrix[i][2][i-1] = (1-success_prob)/2
            #t_matrix[i][2][i+1] = 0
            #west
            t_matrix[i][3][i-grid_dim] = (1-success_prob)/2
            t_matrix[i][3][i+grid_dim] = (1-success_prob)/2
            t_matrix[i][3][i-1] = success_prob
            #t_matrix[i][3][i+1] = 0
        # south border not in corners
        elif i > grid_dim * (grid_dim-1):
            #north 
            t_matrix[i][0][i-grid_dim] = success_prob
            #t_matrix[i][0][i+grid_dim] = 0
            t_matrix[i][0][i-1] = (1-success_prob)/2
            t_matrix[i][0][i+1] = (1-success_prob)/2
            #east
            t_matrix[i][1][i-grid_dim] = (1-success_prob)/2
            #t_matrix[i][1][i+grid_dim] = 0
            t_matrix[i][1][i-1] = (1-success_prob)/2
            t_matrix[i][1][i+1] = success_prob
            #south
            t_matrix[i][2][i-grid_dim] = 1/3
            #t_matrix[i][2][i+grid_dim] = 0
            t_matrix[i][2][i-1] = 1/3
            t_matrix[i][2][i+1] = 1/3
            #west
            t_matrix[i][3][i-grid_dim] = (1-success_prob)/2
            #t_matrix[i][3][i+grid_dim] = 0
            t_matrix[i][3][i-1] = success_prob
            t_matrix[i][3][i+1] = (1-success_prob)/2
        # inner state case
        else:
            #north
            t_matrix[i][0][i-grid_dim] = success_prob
            t_matrix[i][0][i+grid_dim] = (1-success_prob)/3
            t_matrix[i][0][i-1] = (1-success_prob)/3
            t_matrix[i][0][i+1] = (1-success_prob)/3
            #east
            t_matrix[i][1][i-grid_dim] = (1-success_prob)/3
            t_matrix[i][1][i+grid_dim] = (1-success_prob)/3
            t_matrix[i][1][i-1] = (1-success_prob)/3
            t_matrix[i][1][i+1] = success_prob
            #south
            t_matrix[i][2][i-grid_dim] = (1-success_prob)/3
            t_matrix[i][2][i+grid_dim] = success_prob
            t_matrix[i][2][i-1] = (1-success_prob)/3
            t_matrix[i][2][i+1] = (1-success_prob)/3
            #west
            t_matrix[i][3][i-grid_dim] = (1-success_prob)/3
            t_matrix[i][3][i+grid_dim] = (1-success_prob)/3
            t_matrix[i][3][i-1] = success_prob
            t_matrix[i][3][i+1] = (1-success_prob)/3

    # for i in range(n_state):
    #     for j in range(n_action):
    #         # t_matrix[i][j] = t_matrix[i][j]/np.sum(t_matrix[i][j])
    #         sums.append(np.sum(t_matrix[i][j]))
    # #check to make sure valid transition probabilities have been create
    # plt.plot(sums)

    gamma = 0.99

    dataset_len = 4000
    delta_tol = 0.99

    x = np.arange(grid_dim)
    y = np.arange(grid_dim)

    xx, yy = np.meshgrid(x,y)
    xy = np.vstack([yy.reshape(-1),xx.reshape(-1)]).T

    rew_dists = cdist(xy,np.reshape([(grid_dim-1)/2,(grid_dim-1)/2],(1,-1)),metric='sqeuclidean')
    rew_vect = np.exp(-rew_dists/rew_sigma)
    rew_vect2d = np.reshape(rew_vect,(grid_dim,grid_dim))

    return t_matrix, xy, rew_vect, rew_vect2d
    # plt.figure(dpi=100)
    # plt.imshow(rew_vect2d)
    # rewards = np.tile(rew_vect,(1,n_action))

def VI_discrete(t_matrix,rewards,gamma=0.99,theta=0.001,max_iter=10000):

    n_state = rewards.shape[0]
    n_action = rewards.shape[1]

    V_VI = np.zeros(n_state)

    iter = 0

    deltas = []

    err = np.inf

    VI_policy = np.zeros(n_state,dtype=int)
    while iter<max_iter and err>theta:
        if not iter%100:
            print(iter)
        V_temp = np.zeros_like(V_VI)
        buffer = rewards + gamma*t_matrix@V_VI
        V_temp = np.min(buffer,axis=1)
        VI_policy = np.argmin(buffer,axis=1)
        # for state in range(n_state):
        #     # V_prod[labels] = V[labels]
        #     max_buffer = np.zeros(n_action)
        #     for action in range(n_action):
        #         max_buffer[action] = rewards[state][action] + gamma*(np.dot(V_VI,t_matrix[state][action]))
        #     V_temp[state] = np.min(max_buffer)
        #     VI_policy[state] = np.argmin(max_buffer)
        err = np.max(np.abs(V_VI-V_temp))
        deltas.append(err)
        V_VI = V_temp    
        iter = iter+1
    deltas = np.array(deltas)

    return V_VI,VI_policy,deltas

def cycle(my_list, start_at):
    count = 0
    newlst = []
    while count<4 :
        newlst.append(my_list[start_at])
        start_at = (start_at + 1) % len(my_list)
        count += 1
    return newlst

def knn_sigmas(dataset,num_nbrs):

    dataset_len = dataset.shape[0]

    nbrs = NearestNeighbors(n_neighbors= num_nbrs, algorithm='ball_tree').fit(dataset[:,0:2])
    distances_s, indices = nbrs.kneighbors(dataset[:,0:2])
    # print(distances_s.shape)
    # sigma_s = np.mean(distances_s[-1])
    sigma_s = np.mean(distance_s)

    nbrs = NearestNeighbors(n_neighbors= num_nbrs, algorithm='ball_tree').fit(dataset[:,-1].reshape(-1,1))
    distances_a, indices = nbrs.kneighbors(dataset[:,-1].reshape(-1,1))

    # sigma_a = np.mean(distances_a[-1])
    sigma_a = np.mean(distances_a)

    return(sigma_s,sigma_a)

def kw_mean_dist(X,div=1):
    
    dxx = pdist(X, 'euclidean')
    dx = np.median(dxx)
    sigma_st = np.sqrt(dx)
    sigma = sigma_st/div
    gamma = 1/(2*sigma**2)

    return sigma, gamma

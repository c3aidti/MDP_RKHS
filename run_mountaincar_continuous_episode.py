import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import gym
from tqdm.notebook import tqdm
import pickle as pkl

with open("mountaincar_continuous_CME_output.pkl", "rb") as in_file:
    X_fit,X_scaler,actions,W,gamma_k,V = pkl.load(in_file)

env = gym.make('MountainCarContinuous-v0')

print('Number of Data Points: '+str(X_fit.shape[0]))
print('Gamma: '+str(gamma_k))

state = env.reset()
done = False

ep_reward = 0

while True:
    s_rep = np.repeat(state.reshape(1,-1),len(actions),axis=0)
    s_test = X_scaler.transform(np.hstack((s_rep,actions)))

    alphas = (W@rbf_kernel(X_fit,s_test,gamma=gamma_k)).T + 0.0000000001
    alphas = alphas/np.abs(alphas).sum(axis=1,keepdims=True)

    V_out = alphas@V

    action = np.array(actions[int(np.argmin(V_out))])

    env.render()

    state, reward, done, _ = env.step(action)
    ep_reward+=reward
    
    if done:
        print('Episode Reward: '+str(ep_reward))
        if state[0]>=0.5:
            print('Success!')
        break



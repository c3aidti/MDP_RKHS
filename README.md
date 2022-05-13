# Profiling Results

Time-based profiling of the entire application shows that:

- 95.8% of the time is spent on `CME_VI` under [`kernel_MDP.py`](./kernel_MDP.py)

![image](https://user-images.githubusercontent.com/84105092/168286790-aa5e70db-29c4-4527-8735-71541d3c1233.png)

Not surprising, since this function has a `while` loop.

Additional line-by-line profiling into that function shows that:

- 99.6% of the time spent in this function takes place at line 62, doing matrix multipliciation (`alphas@V`)

```plain
Total time: 989.358 s
File: /Users/babreu/c3ai/dahlin/MDP_RKHS/kernel_MDP.py
Function: CME_VI at line 43

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    43                                           @profile
    44                                           def CME_VI(states,rewards,alphas,gamma=0.99,theta=0.001,max_val_iter=10000,err=np.infty,V_init=None):
    45         1         14.0     14.0      0.0      n_state = states.shape[0]
    46         1          1.0      1.0      0.0      if not V_init:
    47         1         32.0     32.0      0.0          V = np.zeros((n_state,1))
    48                                           
    49         1        121.0    121.0      0.0      print('n_state '+str(n_state))
    50                                           
    51         1          0.0      0.0      0.0      val_iter = 0
    52         1          1.0      1.0      0.0      deltas = []
    53         1          4.0      4.0      0.0      deltas_pol = []
    54                                           
    55         1         21.0     21.0      0.0      policy = np.zeros(n_state)
    56                                           
    57      6619      21998.0      3.3      0.0      while val_iter<max_val_iter and err>theta:
    58      6618      10503.0      1.6      0.0          if not val_iter%100:
    59        67       3066.0     45.8      0.0              print('CME_VI iteration '+str(val_iter))
    60        67       1045.0     15.6      0.0              print(err)
    61      6618     197941.0     29.9      0.0          V_temp = np.zeros_like(V)
    62      6618  985055789.0 148844.9     99.6          buffer = rewards + gamma*np.squeeze(alphas@V)
    63      6618    2704378.0    408.6      0.3          V_temp = np.min(buffer,axis=1)
    64                                                   # if not val_iter%100:
    65                                                   #     print(len(np.where(policy==np.argmin(buffer,axis=1))[0]))
    66                                                   # deltas_pol.append(np.max(np.abs(policy-np.argmin(buffer,axis=1))))
    67      6618    1019799.0    154.1      0.1          policy = np.argmin(buffer,axis=1)
    68      6618     310445.0     46.9      0.0          err = np.max(np.abs(V-V_temp))
    69                                                   # print(deltas_pol)
    70                                                   # deltas[val_iter] = err
    71      6618      16406.0      2.5      0.0          deltas.append(err)
    72      6618       8280.0      1.3      0.0          V = V_temp
    73                                                   # V = copy.deepcopy(V_temp)
    74      6618       8461.0      1.3      0.0          val_iter = val_iter+1
    75                                               
    76         1         16.0     16.0      0.0      return V,policy,val_iter,deltas,deltas_pol

```

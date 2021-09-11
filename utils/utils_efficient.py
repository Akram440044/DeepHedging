import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm
keras = tf.keras


def simulate_GBM(m,Ktrain,N,T, mu, sigma,S0, grid_type):
    if grid_type == 'equi':
        time_grid = np.linspace(0,T,N+1)
    elif grid_type == 'exp':
        time_grid = 1.2**np.arange(0, N+1, 1)
        time_grid = (time_grid-1)/time_grid
        time_grid = time_grid/time_grid[-1]*T
        dt = np.diff(time_grid)
    elif grid_type == 'equi-exp':
        N1 = int(N/4)
        N2 = N - N1
        T1 = T/2
        T2 = T/2
        q = 0.97
        a0 = T2*(q-1)/(q**N2-1)
        time_grid1 = np.linspace(0,T1,N1+1)
        time_grid2 = np.cumsum(a0*q**np.arange(0,N2))+T1
        time_grid = np.concatenate([time_grid1,time_grid2])
        time_grid
    dt = np.diff(time_grid)
    BM_path_helper = np.random.normal(size = (Ktrain,N,m))
    BM_path_helper = BM_path_helper * np.sqrt(dt)[:,None] # generate and sum the increment of BM
    BM_path_helper = np.cumsum(BM_path_helper, axis=1) # generate and sum the increment of BM
    BM_path = np.concatenate([np.zeros([Ktrain,1,m]),BM_path_helper],axis = 1) # set initial position of BM be 0 
    price_path = S0 * np.exp(sigma * BM_path +  (mu - 0.5 * sigma **2) * time_grid[None,:,None])  # from BM to geometric BM
    return price_path, time_grid
    

def build_network(m, n, d, N):
    n = m + 15
# architecture is the same for all networks
    Networks = []
    trainable = True
    for j in range(N):
        inputs = keras.Input(shape=(m,))
        x = inputs
        x = keras.layers.BatchNormalization()(x)
        for i in range(d):
            if i < d-1:
                nodes = n
                layer = keras.layers.Dense(nodes, activation='relu',trainable=trainable,
                          kernel_initializer=keras.initializers.RandomNormal(0,1),#kernel_initializer='random_normal',
                          bias_initializer='random_normal',
                          name=str(j) + 'step' + str(i) + 'layer')
                x = layer(x)
#                 x = keras.layers.BatchNormalization()(x)
#                 x = tf.nn.relu(x)
                    
            else:
                nodes = m
                layer = keras.layers.Dense(nodes, activation='sigmoid', trainable=trainable,
                              kernel_initializer=keras.initializers.RandomNormal(0,0.1),#kernel_initializer='random_normal',
                              bias_initializer='random_normal',
                              name=str(j) + 'step' + str(i) + 'layer')
                outputs = layer(x)*10
                network = keras.Model(inputs = inputs, outputs = outputs)
                Networks.append(network)
    return Networks



def BlackScholes(tau, S, K, sigma, option_type):
    d1=np.log(S/K)/sigma/np.sqrt(tau)+0.5*sigma*np.sqrt(tau)
    d2=d1-sigma*np.sqrt(tau)
    delta=norm.cdf(d1) 
    gamma=norm.pdf(d1)/(S*sigma*np.sqrt(tau))
    vega=S*norm.pdf(d1)*np.sqrt(tau)
    theta=-.5*S*norm.pdf(d1)*sigma/np.sqrt(tau)
    if option_type == 'eurodigitalcall':
        price = norm.cdf(d2)
        hedge_strategy = gamma
    else:
        price = (S*norm.cdf(d1)-K*norm.cdf(d2))
        hedge_strategy = delta
    return price, hedge_strategy



def BS0(tau, S, K, sigma):
    K1 = K[0]
    K2 = K[1]
    d1 = (np.log(K1/S) + 0.5*sigma**2*tau) / (sigma*np.sqrt(tau))
    d2 = (np.log(K2/S) + 0.5*sigma**2*tau) / (sigma*np.sqrt(tau))
    d1_prime = d1 - sigma*np.sqrt(tau)
    d2_prime = d2 - sigma*np.sqrt(tau)
    price = S*(norm.cdf(d2_prime) - norm.cdf(d1_prime)) - K1*(norm.cdf(d2) - norm.cdf(d1))
    hedge_strategy = (norm.cdf(d2_prime) - norm.cdf(d1_prime)) - (norm.pdf(d2_prime) - norm.pdf(d1_prime))/(sigma*np.sqrt(tau))\
    + (K1/S)*(norm.pdf(d2) - norm.pdf(d1))/(sigma*np.sqrt(tau))
    return price, hedge_strategy

def BS1(tau, S, K, sigma):
    K1 = K[0]
    K2 = K[1]
    d1 = np.log(S/K2)/sigma/np.sqrt(tau) + 0.5*sigma*np.sqrt(tau)
    d2 = d1-sigma*np.sqrt(tau)
    price = S*norm.cdf(d1) - K1*norm.cdf(d2)
    hedge_strategy = norm.cdf(d1) + (norm.pdf(d1) - K1/S*norm.pdf(d2))/(sigma*np.sqrt(tau))
    return price, hedge_strategy


def BSinf(tau, S, K, sigma):
    K1 = K[0]
    K2 = K[1]
    d1 = np.log(S/K2)/sigma/np.sqrt(tau)+0.5*sigma*np.sqrt(tau)
    d2 = d1-sigma*np.sqrt(tau)
    price = (S*norm.cdf(d1)-K2*norm.cdf(d2))
    hedge_strategy = norm.cdf(d1)
    return price, hedge_strategy

    
def delta_hedge(price_path,payoff,T,K,sigma,po,time_grid):
    price = price_path
    batch, N, m = price.shape
    N = N - 1 
    price_difference = price[:,1:,:] - price[:,:-1,:]  
    
    hedge_path = np.zeros_like(price)
    option_path = np.zeros_like(price) 
    
    if po == 0:
        BS_func = BS0
    elif po == 1:
        BS_func = BS1
    elif po == np.inf:
        BS_func = BSinf   
    else:
        print(po)
        print('ERROR')
        
    premium,_ = BS_func(T-time_grid[0], price[:,0,:], K, sigma)
    hedge_path[:,0,:] =  premium
    option_path[:,-1,:] =  payoff
    
    for j in range(N):
        option_price, strategy = BS_func(T-time_grid[j],price[:,j,:],K,sigma)  
        hedge_path[:,j+1] = hedge_path[:,j] + strategy * price_difference[:,j,:]   
        option_path[:,j,:] =  option_price
        
    outputs = hedge_path[:,-1] 
    return outputs, hedge_path , option_path
    
    
def delta_hedge_cost(price_path,payoff,T,K,sigma,po,time_grid):
    price = price_path
    batch, N, m = price.shape
    N = N - 1 
    price_difference = price[:,1:,:] - price[:,:-1,:]  
    hedge_path = np.zeros_like(price)
    option_path = np.zeros_like(price) 
    if po == 0:
        BS_func = BS0
    elif po == 1:
        BS_func = BS1
    elif po == np.inf:
        BS_func = BSinf   
    else:
        print('ERROR')
    premium,_ = BS_func(T-time_grid[0], price[:,0,:], K, sigma)
    hedge_path[:,0,:] =  premium
    option_path[:,-1,:] =  payoff
    STRATEGY = []
    for j in range(N):
        option_price, strategy = BS_func(T-time_grid[j],price[:,j,:],K,sigma)  
        STRATEGY.append(strategy)
        cost = 0
        if j > 0: 
            cost = 0.001*((STRATEGY[j]- STRATEGY[j-1])*price[:,j,:])**2
        hedge_path[:,j+1] = hedge_path[:,j] + strategy * price_difference[:,j,:] - cost 
        option_path[:,j,:] =  option_price
        
    outputs = hedge_path[:,-1] 
    return outputs, hedge_path , option_path

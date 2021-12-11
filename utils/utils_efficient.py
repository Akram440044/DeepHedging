import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm
keras = tf.keras
tf.keras.backend.set_floatx('float64')


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
    path_1 = sigma * BM_path +  mu * time_grid[None,:,None]
    path_2 = price_path*0 - sigma**2/2 * time_grid[None,:,None]
    return price_path, time_grid, path_1, path_2
    

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
                outputs = layer(x)*2
                network = keras.Model(inputs = inputs, outputs = outputs)
                Networks.append(network)
    return Networks



def BlackScholes(tau, S, K, sigma, option_type = 'eurodigitalcall'):
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



def BS0(tau, S, K,L, mu,sigma,p):
    K1 = K
    K2 = L
    d1 = (np.log(K1/S) + 0.5*sigma**2*tau) / (sigma*np.sqrt(tau))
    d2 = (np.log(K2/S) + 0.5*sigma**2*tau) / (sigma*np.sqrt(tau))
    d1_prime = d1 - sigma*np.sqrt(tau)
    d2_prime = d2 - sigma*np.sqrt(tau)
    price = S*(norm.cdf(d2_prime) - norm.cdf(d1_prime)) - K1*(norm.cdf(d2) - norm.cdf(d1))
    hedge_strategy = (norm.cdf(d2_prime) - norm.cdf(d1_prime)) - (norm.pdf(d2_prime) - norm.pdf(d1_prime))/(sigma*np.sqrt(tau))\
    + (K1/S)*(norm.pdf(d2) - norm.pdf(d1))/(sigma*np.sqrt(tau))
    return price, hedge_strategy

def BS1(tau, S, K,L, mu,sigma,p):
    K1 = K
    K2 = L
    d1 = np.log(S/K2)/sigma/np.sqrt(tau) + 0.5*sigma*np.sqrt(tau)
    d2 = d1-sigma*np.sqrt(tau)
    price = S*norm.cdf(d1) - K1*norm.cdf(d2)
    hedge_strategy = norm.cdf(d1) + (norm.pdf(d1) - K1/S*norm.pdf(d2))/(sigma*np.sqrt(tau))
    return price, hedge_strategy


import tensorflow_probability as tfp
tfd = tfp.distributions
a = tf.cast(0,tf.float64)
b = tf.cast(1,tf.float64)
dist = tfd.Normal(loc=a, scale=b)
def BS1_tf(tau, S, K,L, mu,sigma,p):
    K1 = K
    K2 = L
    d1 = tf.math.log(S/K2)/sigma/tf.math.sqrt(tau) + 0.5*sigma*tf.math.sqrt(tau)
    d2 = d1-sigma*np.math.sqrt(tau)
#     price = S*dist.cdf(d1) - K1*dist.cdf(d2)
    hedge_strategy = dist.cdf(d1) + (dist.prob(d1) - K1/S*dist.prob(d2))/(sigma*tf.math.sqrt(tau))
    return hedge_strategy

def BSinf(tau, S, K,L, mu,sigma,p):
    K1 = K
    K2 = L
    d1 = np.log(S/K2)/sigma/np.sqrt(tau)+0.5*sigma*np.sqrt(tau)
    d2 = d1-sigma*np.sqrt(tau)
    price = (S*norm.cdf(d1)-K2*norm.cdf(d2))
    hedge_strategy = norm.cdf(d1)
    return price, hedge_strategy

def BSinf_tf(tau, S, K,L, mu,sigma,p):
#     K1 = K ## Not useful
    K2 = L 
    d1 = tf.math.log(S/K2)/sigma/tf.math.sqrt(tau)+0.5*sigma*tf.math.sqrt(tau)
    d2 = d1-sigma*tf.math.sqrt(tau)
#     price = (S*dist.cdf(d1)-K2*dist.cdf(d2))
    hedge_strategy = dist.cdf(d1)
    return hedge_strategy


def BSp0(tau, S, K,L, mu,sigma,p):
    K_array = np.array([K,L[0],L[1]])
    b = np.log(K_array/S)/sigma/np.sqrt(tau)-0.5*sigma*np.sqrt(tau)
    d = b+sigma*np.sqrt(tau)
    price = S*(1 - norm.cdf(b[0]) + norm.cdf(b[1]) - norm.cdf(b[2])) - K*(1 - norm.cdf(d[0]) + norm.cdf(d[1]) - norm.cdf(d[2]))
    hedge_strategy = 1 - norm.cdf(b[0]) + norm.cdf(b[1]) - norm.cdf(b[2])
    return price, hedge_strategy

    
def delta_hedge(price_path,payoff,T,K,L,mu,sigma,po,time_grid, path_1, path_2):
    price = price_path
    batch, N, m = price.shape
    N = N - 1 
    price_difference = price[:,1:,:] - price[:,:-1,:]  
    path_1_diff = path_1[:,1:,:] - path_1[:,:-1,:]  
    path_2_diff = path_2[:,1:,:] - path_2[:,:-1,:]  
    
    hedge_path = np.zeros_like(price)
    option_path = np.zeros_like(price) 
    
    if po == 0:
        BS_func = BS0
    elif po == 1:
        BS_func = BS1
    elif po == np.inf:
        BS_func = BSinf   
    else:
        BS_func = BSp
    bound = 100    
    premium,_ = BS_func(T-time_grid[0], price[:,0,:], K,L, mu,sigma,po) 
    hedge_path[:,0,:] =  premium + bound
    option_path[:,-1,:] =  payoff
    
    for j in range(N):
        option_price, strategy = BS_func(T-time_grid[j],price[:,j,:],K,L,mu,sigma,po)  
#         hedge_path[:,j+1] = hedge_path[:,j] + strategy * price_difference[:,j,:]   
        option_path[:,j,:] =  option_price
        
        hedge1 = hedge_path[:,j] + strategy * price_difference[:,j,:]   

        pi = hedge_path[:,j]*price[:,j,:]/(hedge_path[:,j] + 1e-10)
        dlogV = pi * path_1_diff[:,j,:] + pi**2 * path_2_diff[:,j,:]
        hedge2 = hedge_path[:,j] * np.exp(dlogV)

        ind0 = hedge_path[:,j] == 0
        ind1 = hedge1 >= 0
        ind2 = hedge1 < 0
        hedge_path[:,j+1] = hedge_path[:,j]*ind0 + hedge1*ind1*(1-ind0) + hedge2*ind2*(1-ind0)
        
        
        
    outputs = hedge_path[:,-1] - bound
    return outputs, hedge_path , option_path
    
    
def delta_hedge_cost(price_path,payoff,T,K,L,mu,sigma,po,time_grid, path_1, path_2):
    price = price_path
    batch, N, m = price.shape
    N = N - 1 
    
    price_difference = price[:,1:,:] - price[:,:-1,:]  
    path_1_diff = path_1[:,1:,:] - path_1[:,:-1,:]  
    path_2_diff = path_2[:,1:,:] - path_2[:,:-1,:] 
    hedge_path = np.zeros_like(price)
    option_path = np.zeros_like(price) 
    if po == 0:
        BS_func = BS0
    elif po == 1:
        BS_func = BS1
    elif po == np.inf:
        BS_func = BSinf 
    else:
        BS_func = BSp
    bound = 100
    premium,_ = BS_func(T-time_grid[0], price[:,0,:], K,L, mu,sigma,po)
    hedge_path[:,0,:] =  premium + bound
    option_path[:,-1,:] =  payoff
    STRATEGY = []
    COST = 0
    for j in range(N):
        option_price, strategy = BS_func(T-time_grid[j],price[:,j,:],K,L,mu,sigma,po)  
        hedge1 = hedge_path[:,j] + strategy * price_difference[:,j,:]   

        pi = hedge_path[:,j]*price[:,j,:]/(hedge_path[:,j] + 1e-10)
        dlogV = pi * path_1_diff[:,j,:] + pi**2 * path_2_diff[:,j,:]
        hedge2 = hedge_path[:,j] * np.exp(dlogV)

        ind0 = hedge_path[:,j] == 0
        ind1 = hedge1 >= 0
        ind2 = hedge1 < 0
        hedge_path[:,j+1] = hedge_path[:,j]*ind0 + hedge1*ind1*(1-ind0) + hedge2*ind2*(1-ind0)
       
        STRATEGY.append(strategy)
        cost = 0
        if j > 0: 
            cost = 0.01*tf.math.abs((STRATEGY[j]- STRATEGY[j-1])*price[:,j,:])
        COST += cost 
        option_path[:,j,:] =  option_price
        
    outputs = hedge_path[:,-1] - COST - bound
    return outputs, hedge_path , option_path

alpha = 4
def build_dynamic_cost(m, N, trans_cost, initial_wealth, ploss, po, time_grid,K,L,mu,sigma,T):
    m = 1
    N = 100
    trans_cost = False
    initial_wealth = tf.cast(initial_wealth,tf.float64)
    tau = tf.cast(T-time_grid,tf.float64)
    L = tf.cast(L,tf.float64)
    K = tf.cast(K,tf.float64)
    mu = tf.cast(mu,tf.float64)
    sigma = tf.cast(sigma,tf.float64)

    L_n = 3 # number of layers in strategy
    n = m + 20  # nodes in the first but last layers
    Networks = build_network(m, n , L_n, N)
    Network0 = keras.layers.Dense(1, use_bias=False)

    L_n = 3 # number of layers in strategy
    n = m + 20  # nodes in the first but last layers
    price = keras.Input(shape=(N+1,m)) 
    path_1 = keras.Input(shape=(N+1,m))   #\mu t + \sigma W_t  t=0,..,N+1; (batch, N+1, m)
    path_2 = keras.Input(shape=(N+1,m))   # -\sigma^2/2 t; t=0,..,N+1; (batch, N+1, m)
    payoff = keras.Input(shape=(1))

    inputs = [price, path_1, path_2, payoff]
    price_diff = price[:,1:,:] - price[:,:-1,:]
    path_1_diff = path_1[:,1:,:] - path_1[:,:-1,:]  # \mu dt + \sigma dW_t; t=0,..,N; (batch, N, m)
    path_2_diff = path_2[:,1:,:] - path_2[:,:-1,:]  # -\sigma^2/2 dt; t=0,..,N; (batch, N, m)

    premium = initial_wealth
    HEDGE = [None]*(N+1) 
    bound = 100
#     bound = 0
    HEDGE[0] = tf.zeros_like(price[:,0,:]) + initial_wealth + bound # Wealth process V_t; t=0,..,N+1; (batch, N+1, m)
    STRATEGY = [None]*N  # holding process \theta_t; t=0,..,N; (batch, N, m)
    ADMISSIBLE = tf.zeros_like(price[:,0,:])
    cost_all = 0
    for j in range(N):
        log_price = tf.math.log(price[:,j,:])      
        I = log_price
#         delta = BSinf_tf(tau[j],price[:,j,:],K,L,mu,sigma,1)
#         I = tf.concat([log_price, delta],axis = 1)
#         I  = tf.concat([log_price, HEDGE[j]],axis = 1)
        STRATEGY[j] = Networks[j](I) 
#         HEDGE[j+1] = (1-STRATEGY[j])*HEDGE[j] + STRATEGY[j]*HEDGE[j]*tf.math.exp(path_1_diff[:,j,:] + path_2_diff[:,j,:])
    
#         STRATEGY[j] = BS1_tf(tau[j],price[:,j,:],K,L,mu,sigma,1)

        hedge1 = HEDGE[j] + STRATEGY[j] * price_diff[:,j,:]

        pi = STRATEGY[j]*price[:,j,:]/(HEDGE[j] + 1e-10)
        dlogV = pi * path_1_diff[:,j,:] + pi**2 * path_2_diff[:,j,:]
        hedge2 = HEDGE[j] * tf.math.exp(dlogV)

        ind0 = tf.cast(HEDGE[j] == 0, tf.float64)
        ind1 = tf.cast(hedge1 >= 0, tf.float64)
        ind2 = tf.cast(hedge1 < 0, tf.float64)
        HEDGE[j+1] = HEDGE[j]*ind0 + hedge1*ind1*(1-ind0) + hedge2*ind2*(1-ind0)
        
        cost = 0
        if trans_cost and j > 0: 
            cost = 0.01*tf.math.abs((STRATEGY[j]- STRATEGY[j-1])*price[:,j,:])*(1-ind0)
            cost_all += cost
        ADMISSIBLE = tf.math.minimum(ADMISSIBLE, HEDGE[j+1]-bound)

    outputs = tf.math.reduce_sum(HEDGE[-1],axis = -1, keepdims = True) - bound
    model_hedge = keras.Model(inputs = inputs, outputs=outputs)

# Define LOSS
    loss1 = ploss(payoff, outputs)
    loss1 = tf.reduce_mean(loss1)
    if po not in [0,np.inf]:
        loss1 = loss1 ** (1/po)
    model_hedge.add_loss(loss1) 
    model_hedge.add_metric(loss1, name='p-loss')

    loss2 = tf.nn.relu(-ADMISSIBLE)*alpha
    loss2 = tf.reduce_mean(loss2)
    model_hedge.add_loss(loss2) 
    model_hedge.add_metric(loss2, name='0-ad-loss')
    
#     if trans_cost:
#         loss_cost = tf.reduce_mean(cost_all)
#         model_hedge.add_metric(loss_cost, name='tran_cost')
 
 
    return model_hedge, Network0, Networks

    

def BSp(tau,S0,strike,L,mu,sigma,p):
    x = S0
    K = strike
    alpha = mu/sigma**2
    d1 = np.log(x/L)/sigma/np.sqrt(tau)+0.5*sigma*np.sqrt(tau)
    d2 = d1-sigma*np.sqrt(tau)
    tmp1 = x*norm.cdf(d1) - strike*norm.cdf(d2)
    tmp2 = (L/x)**(alpha/(p-1))*(L-strike)
    tmp3 = 0.5*sigma**2*tau**alpha/(p-1)*(alpha/(p-1)+1)
    tmp4 = d2 - alpha*sigma*np.sqrt(tau)/(p-1)
    price = tmp1  - tmp2 * np.exp(tmp3) * norm.cdf(tmp4) 
    beta = alpha/(p-1)
    tmp1 = norm.cdf(d1)
    tmp2 = beta * (L/x)**beta * (L-K)/x
    tmp3 = tmp3
    tmp4 = tmp4
    delta = tmp1 + tmp2 * np.exp(tmp3) * norm.cdf(tmp4) 
    return price, delta
    
    
def fair_price(L,tau,S0,strike,mu,sigma,p):
    if p == 0:
        BS_func = BS0
        price,_ = BS_func(tau,S0,strike,L,mu,sigma,p)
    elif p == 1:
        BS_func = BS1
        price,_ = BS_func(tau,S0,strike,L,mu,sigma,p)
    elif p == np.inf:
        BS_func = BSinf   
        price,_ = BS_func(tau,S0,strike,L,mu,sigma,p)
    else:
        price,_ = BSp(tau,S0,strike,L,mu,sigma,p)
    return price 

def solver(tau,S0,strike,mu,sigma,p,endow):
    L = np.linspace(strike,strike+200,1000)
    y = fair_price(L,tau,S0,strike,mu,sigma,p)
    idx = np.argmin(np.abs(y-endow))
    print(idx)
    print(y[idx])    
    return L[idx]
    
    
    
    
    
    



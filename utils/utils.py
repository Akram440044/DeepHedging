import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm
keras = tf.keras

def BlackScholes(tau, S, K, sigma, option_type):
    d1=np.log(S/K)/sigma/np.sqrt(tau)+0.5*sigma*np.sqrt(tau)
    d2=d1-sigma*np.sqrt(tau)
    delta=norm.cdf(d1) 
    gamma=norm.pdf(d1)/(S*sigma*np.sqrt(tau))
    vega=S*norm.pdf(d1)*np.sqrt(tau)
    theta=-.5*S*norm.pdf(d1)*sigma/np.sqrt(tau)
    if option_type == 'eurocall':
        price = (S*norm.cdf(d1)-K*norm.cdf(d2))
        hedge_strategy = delta
    elif option_type == 'eurodigitalcall':
        price = norm.cdf(d2)
        hedge_strategy = gamma
    return price, hedge_strategy

import QuantLib as ql

def heston_value(calculation_date, v0, kappa, theta, sigma, rho, spot_price, strike_price, maturity_date ):
    
    # option parameters
    option_type = ql.Option.Call
    day_count = ql.Actual365Fixed()
    calendar = ql.UnitedStates()

    # construct the European Option
    payoff = ql.PlainVanillaPayoff(option_type, strike_price)
    exercise = ql.EuropeanExercise(maturity_date)
    european_option = ql.VanillaOption(payoff, exercise)
    
    # set the evaluation date
    ql.Settings.instance().evaluationDate = calculation_date

    # construct the yield curve
    dividend_rate =  0
    risk_free_rate = 0
    flat_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(calculation_date, risk_free_rate, day_count)
    )
    dividend_yield = ql.YieldTermStructureHandle(
        ql.FlatForward(calculation_date, dividend_rate, day_count)
    )

    # set the spot price
    spot_handle = ql.QuoteHandle(
        ql.SimpleQuote(spot_price)
    )
    
    # calculate option price
    heston_process = ql.HestonProcess(
    flat_ts, dividend_yield, spot_handle,
    v0, kappa, theta, sigma, rho
    )

    engine = ql.AnalyticHestonEngine(ql.HestonModel(heston_process),0.01, 1000)
    european_option.setPricingEngine(engine)
    h_price = european_option.NPV()
    
    return h_price

def simulate_Heston(Ktrain,N,T,rho,kappa,theta,sigma,S0,v0):
    v0 = theta # historical vols for the stock  

    day_count = ql.Actual365Fixed()
    calculation_date = ql.Date(1, 10, 2020)
    spot_price = 1.00
    # ql.Settings.instance().evaluationDate = calculation_date

    # construct the yield curve
    dividend_rate =  0
    risk_free_rate = 0
    flat_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(calculation_date, risk_free_rate, day_count)
    )
    dividend_yield = ql.YieldTermStructureHandle(
        ql.FlatForward(calculation_date, dividend_rate, day_count)
    )

    # set the spot price
    spot_handle = ql.QuoteHandle(
        ql.SimpleQuote(spot_price)
    )

    # calculate option price
    heston_process = ql.HestonProcess(
    flat_ts, dividend_yield, spot_handle,
    v0, kappa, theta, sigma, rho
    )


    timestep = N
    length = T
    times = ql.TimeGrid(length, timestep)
    dimension = heston_process.factors()

    rng = ql.UniformRandomSequenceGenerator(dimension * timestep, ql.UniformRandomGenerator())
    sequenceGenerator = ql.GaussianRandomSequenceGenerator(rng)
    pathGenerator = ql.GaussianMultiPathGenerator(heston_process, list(times), sequenceGenerator, False)

    # paths[0] will contain spot paths, paths[1] will contain vol paths
    paths = [[] for i in range(dimension)]
    for i in range(Ktrain):
        samplePath = pathGenerator.next()
        values = samplePath.value()
        spot = values[0]

        for j in range(dimension):
            paths[j].append([x for x in values[j]])


    price_path, vol_path = np.array(paths)[0,:,:], np.array(paths)[1,:,:]
    price_path = price_path[:,:,None]
    vol_path = vol_path[:,:,None]


    
    return price_path, vol_path

# def stimulate_Heston(m,Ktrain,N,T,corr,kappa,theta,xi,S0,v0):
#     S = [S0]
#     v = [v0]
#     v_curr = v0; S_curr = S0
#     for i in range(N):
#         # Simulate standard normal random variables
#         Z1_i = np.random.normal(0,1,(Ktrain,m))
#         Z2_i = np.random.normal(0,1,(Ktrain,m))
#         # Generate correlated Brownian motions
#         WS_curr = np.sqrt(T/N)*Z1_i
#         Wv_curr = np.sqrt(T/N)*(corr*Z1_i + np.sqrt(1-corr**2)*Z2_i)
#         # Calculate volatility and stock price
#         # Adjustment: np.abs(v_curr) for np.sqrt() in the last term
# #         v_new = v_curr + kappa*(theta-v_curr)*T/N + xi*np.sqrt(np.abs(v_curr))*Wv_curr
#         v_new = v_curr + kappa*(theta-v_curr)*T/N + xi*np.sqrt(v_curr)*Wv_curr # without abs 
#         S_new = S_curr * np.exp(- v_new**2/2*T/N + v_new*WS_curr)
#         v_curr = v_new; S_curr = S_new
#         # Append the results to the arrays of each day's value
#         v += [v_new]; S += [S_new]
#         price_path = np.swapaxes(np.array(S),0,1)
#         vol_path = np.swapaxes(np.array(v),0,1)
#     return price_path, vol_path


# def stimulate_GBM(m,Ktrain,N,T,sigma,S0):
#     time_grid = np.linspace(0,T,N+1)
#     dt = T/N
#     BM_path_helper = np.cumsum(np.random.normal(size = [Ktrain,N,m], loc=0, scale=np.sqrt(dt)),axis = 1) # generate and sum the increment of BM
#     BM_path = np.concatenate([np.zeros([Ktrain,1,m]),BM_path_helper],axis = 1) # set initial position of BM be 0 
#     price_path = S0 * np.exp(sigma * BM_path - 0.5 * sigma **2 * time_grid[None,:,None])  # from BM to geometric BM
#     return price_path, time_grid


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
        T1 = 0.5
        T2 = 0.5
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
# architecture is the same for all networks
    Networks = []
    trainable = True
    for j in range(N):
        inputs = keras.Input(shape=(m,))
        x = inputs
#         x = keras.layers.BatchNormalization()(x)
        for i in range(d):
            if i < d-1:
                nodes = n
                layer = keras.layers.Dense(nodes, activation='linear',trainable=trainable,
                          kernel_initializer=keras.initializers.RandomNormal(0,1),#kernel_initializer='random_normal',
                          bias_initializer='random_normal',
                          name=str(j) + 'step' + str(i) + 'layer')
                x = layer(x)
#                 x = keras.layers.BatchNormalization()(x)
                x = tf.nn.relu(x)
                    
            else:
                nodes = m
                layer = keras.layers.Dense(nodes, activation='linear', trainable=trainable,
                              kernel_initializer=keras.initializers.RandomNormal(0,0.1),#kernel_initializer='random_normal',
                              bias_initializer='random_normal',
                              name=str(j) + 'step' + str(i) + 'layer')
                outputs = layer(x)
                network = keras.Model(inputs = inputs, outputs = outputs)
                Networks.append(network)
    return Networks


def delta_hedge(price_path,payoff, T,K,sigma,option_type,time_grid):
    price = price_path
    batch, N, m = price.shape
    N = N - 1 
#     time_grid = np.linspace(0,T,N+1)
    price_difference = price[:,1:,:] - price[:,:-1,:]  
    
    hedge_path = np.zeros_like(price)
    option_path = np.zeros_like(price) 
    premium,_ = BlackScholes(T-time_grid[0], price[:,0,:], K, sigma, option_type)
    
    hedge_path[:,0,:] =  premium
    option_path[:,-1,:] =  payoff
    
    for j in range(N):
        option_price, strategy = BlackScholes(T-time_grid[j],price[:,j,:],K,sigma,option_type)  
        hedge_path[:,j+1] = hedge_path[:,j] + strategy * price_difference[:,j,:]   
        option_path[:,j,:] =  option_price
    outputs = hedge_path[:,-1] 
    return outputs, hedge_path , option_path
    
    


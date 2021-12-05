#美式期权CRR
import numpy as np
import math

def CRR_option_valuation_Am(S0, T, r, K, sigma, M=500):
    ''' Cox-Ross-Rubinstein American option valuation.

    Parameters
    ==========
    S0 : float
        stock/index level at time 0
    K : float
        strike price
    T : float
        date of maturity
    r : float
        constant, risk-less short rate
    sigma : float
        volatility
    type : string
        either 'call' or 'put'
    M : int
        number of time intervals
    Returns
    =======
    price : float
        American option present value at t
    '''
    dt = T / M  # time interval
    df = math.exp(-r * dt)  # discount factor
    u = math.exp(sigma * math.sqrt(dt))  # up-movement
    d = 1 / u  # down-movement
    q = (math.exp(r * dt) - d) / (u - d)  # martingale probability
    # Array Generation for Stock Prices
    mu = np.arange(M + 1)
    mu = np.resize(mu, (M + 1, M + 1))
    md = np.transpose(mu)
    mu = u ** (mu - md)
    md = d ** md
    S = S0 * mu * md

    # Valuation by Backwards Induction
    h = np.maximum(K - S, 0)  # innver value matrix
    V = np.maximum(K - S, 0)  # value matrix
    C = np.zeros((M + 1, M + 1), dtype=np.float)  # continuation values
    ex = np.zeros((M + 1, M + 1), dtype=np.float)  # exercise matrix

    z = 0
    for i in range(M - 1, -1, -1):
        C[0:M - z, i] = (q * V[0:M - z, i + 1] +
                         (1 - q) * V[1:M - z + 1, i + 1]) * df
        V[0:M - z, i] = np.where(h[0:M - z, i] > C[0:M - z, i],
                                 h[0:M - z, i], C[0:M - z, i])
        ex[0:M - z, i] = np.where(h[0:M - z, i] > C[0:M - z, i], 1, 0)
        z += 1
    price = V[0, 0]
    return price
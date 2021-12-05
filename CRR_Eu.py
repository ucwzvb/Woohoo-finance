import numpy as np
import math
def CRR_option_value(S0, K, T, r, sigma, type, M):
    ''' Cox-Ross-Rubinstein European option valuation.

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
        European option present value at t
    '''
    # Time Parameters
    dt = T / M  # length of time interval
    df = math.exp(-r * dt)  # discount per interval

    # Binomial Parameters
    u = math.exp(sigma * math.sqrt(dt))  # up movement
    d = 1 / u  # down movement
    q = (math.exp(r * dt) - d) / (u - d)  # martingale branch probability

    # Array Initialization for Index Levels
    mu = np.arange(M + 1)
    mu = np.resize(mu, (M + 1, M + 1))
    md = np.transpose(mu)
    mu = u ** (mu - md)
    md = d ** md
    S = S0 * mu * md

    # Inner Values
    try:
        if type == 'C':
            V = np.maximum(S - K, 0)  # inner values for European call option
        elif type == 'P':
            V = np.maximum(K - S, 0)  # inner values for European put option
    except:
        print('please confrim all the paramnetars')
    z = 0
    for t in range(M - 1, -1, -1):  # backwards iteration
        V[0:M - z, t] = (q * V[0:M - z, t + 1] +
                         (1 - q) * V[1:M - z + 1, t + 1]) * df
        z += 1
    price = V[0, 0]
    return price
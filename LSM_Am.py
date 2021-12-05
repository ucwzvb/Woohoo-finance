# 美式期权LSM_primal
import numpy as np
from scipy.stats import norm
from scipy.stats.stats import sigmaclip
from scipy import stats

def LSM(S0, K, T, r, sigma, I, M):
    ''' Longstaff-Schwartz American option valuation.

    Parameters
    ==========
    S0 : float
        stock/index level at t=0
    K : float
        strike price
    T : float
        date of maturity
    r : float
        constant, risk-less short rate
    sigma : float
        volatility
    I : int
        number of time intervals
    M : int
        number of paths

    Returns
    =======
    price : float
        American option present value at t
    '''
    dt = T / M  # time interval
    df = np.exp(-r * dt)  # discount factor per time time interval

    S = S0 * np.exp(np.cumsum((r - 0.5 * sigma ** 2) * dt +
                              sigma * np.sqrt(dt) *
                              np.random.standard_normal((M + 1, I)), axis=0))
    S[0] = S0

    # Inner Values
    h = np.maximum(K - S, 0)

    # Present Value Vector (Initialization)
    V = h[-1]

    # American Option Valuation by Backwards Induction
    for t in range(M - 1, 0, -1):
        rg = np.polyfit(S[t], V * df, 5)
        C = np.polyval(rg, S[t])  # continuation values
        V = np.where(h[t] > C, h[t], V * df)
        # exercise decision
    V0 = df * np.sum(V) / I  # LSM estimator
    price = V0
    return price
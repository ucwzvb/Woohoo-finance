import numpy as np
from scipy.stats import norm

def BSM_Eu(S,K,r,T,sigma,type):
    ''' Calculates Black-Scholes-Merton European call/put option value.

    Parameters
    ==========
    S : float
        stock/index level at time t
    K : float
        strike price
    T : float
        date of maturity
    r : float
        constant, risk-less short rate
    sigma : float
        volatility
    type : string
        call/put option

    Returns
    =======
    price : float
        European option present value at t
    '''
    d1 = (np.log(S/K)+(r+sigma**2/2)*(T))/(sigma*np.sqrt(T))
    d2 = d1-sigma*np.sqrt(T)
    try:
        if type == 'C':
            price = S*norm.cdf(d1,0,1)-K*np.exp(-r*(T))*norm.cdf(d2,0,1)
        elif type == 'P':
            price = K*np.exp(-r*(T))*norm.cdf(-d2, 0, 1)- S*norm.cdf(-d1,0,1)
        return price
    except:
        print('please confrim all the paramnetars')
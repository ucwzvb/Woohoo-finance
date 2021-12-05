from random import gauss
from math import exp, sqrt


def calculate_S_T(S, theta, r, T):
    """calculate stock present value at t"""
    return S * exp((r - 0.5 * theta ** 2) * T + theta * sqrt(T) * gauss(0.0, 1.0))


def option_payoff(type, S, K):
    """calculate stock present value at t"""
    try:
        if type == 'C':
            return max(S - K, 0.0)  # values for European call option
        elif type == 'P':
            return max(K - S, 0.0)  # values for European put option
    except:
        print('please confrim all the paramnetars')


def mcs_option_price(S, K, T, r, sigma, type, simulations):
    ''' Monte Carlo Simulation European option valuation.

    Parameters
    ==========
    S : float
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
    simulations : int
        number of time simulations
    Returns
    =======
    price : float
        European option present value at t
    '''
    payoffs = []
    discout = exp(-r * T)
    for i in range(simulations):
        S_T = calculate_S_T(S, sigma, r, T)
        payoffs.append(
            option_payoff(type, S_T, K)
        )
    price = discout * sum(payoffs) / float(simulations)
    return price
#FFT
import numpy as np
from numpy.fft import fft
from scipy.integrate import quad
from scipy import stats
def BSM_call_value_INT(S0, K, T, r, sigma):
    ''' Fourier-based approach (integral).

    Parameters
    ==========
    S0: float
        initial stock/index level
    K: float
        strike price
    T: float
        time-to-maturity (for t=0)
    r: float
        constant risk-free short rate
    sigma: float
        volatility factor in diffusion term

    Returns
    =======
    call_value: float
        European call option present value
    '''
    int_value = quad(lambda u:
                     BSM_integral_function(u, S0, K, T, r, sigma), 0, 100)[0]
    call_value = max(0, S0 - np.exp(-r * T) * np.sqrt(S0 * K) /
                     np.pi * int_value)
    return call_value


def BSM_integral_function(u, S0, K, T, r, sigma):
    ''' Fourier-based approach: integral function. '''
    cf_value = BSM_characteristic_function(u - 1j * 0.5, 0.0, T, r, sigma)
    int_value = 1 / (u ** 2 + 0.25) \
        * (np.exp(1j * u * np.log(S0 / K)) * cf_value).real
    return int_value


def BSM_characteristic_function(v, x0, T, r, sigma):
    '''Fourier-based approach: charcteristic function. '''
    cf_value = np.exp(((x0 / T + r - 0.5 * sigma ** 2) * 1j * v -
                       0.5 * sigma ** 2 * v ** 2) * T)
    return cf_value


def BSM_call_value_FFT(S0, K, r, T, sigma):
    '''Fourier-based approach (integral).

    Parameters
    ==========
    S0: float
        initial stock/index level
    K: float
        strike price
    T: float
        time-to-maturity (for t=0)
    r: float
        constant risk-free short rate
    sigma: float
        volatility factor in diffusion term

    Returns
    =======
    call_value: float
        European call option present value

    '''
    k = np.log(K / S0)
    x0 = np.log(S0 / S0)
    g = 1  # factor to increase accuracy
    N = g * 4096
    eps = (g * 150.) ** -1
    eta = 2 * np.pi / (N * eps)
    b = 0.5 * N * eps - k
    u = np.arange(1, N + 1, 1)
    vo = eta * (u - 1)
    # Modificatons to Ensure int_valueegrability
    if S0 >= 0.95 * K:  # ITM case
        alpha = 1.5
        v = vo - (alpha + 1) * 1j
        modcharFunc = np.exp(-r * T) * (BSM_characteristic_function(
            v, x0, T, r, sigma) /
            (alpha ** 2 + alpha - vo ** 2 + 1j * (2 * alpha + 1) * vo))
    else:        # OTM case
        alpha = 1.1
        v = (vo - 1j * alpha) - 1j
        modcharFunc1 = np.exp(-r * T) * (1 / (1 + 1j * (vo - 1j * alpha)) -
                             np.exp(r * T) / (1j * (vo - 1j * alpha)) -
                             BSM_characteristic_function(v, x0, T, r, sigma) /
                             ((vo - 1j * alpha) ** 2 -
                              1j * (vo - 1j * alpha)))
        v = (vo + 1j * alpha) - 1j
        modcharFunc2 = np.exp(-r * T) * (1 / (1 + 1j * (vo + 1j * alpha))
                             - np.exp(r * T) /
                             (1j * (vo + 1j * alpha))
                             - BSM_characteristic_function(
            v, x0, T, r, sigma) /
            ((vo + 1j * alpha) ** 2 - 1j * (vo + 1j * alpha)))
    # Numerical FFT Routine
    delt = np.zeros(N, dtype=np.float)
    delt[0] = 1
    j = np.arange(1, N + 1, 1)
    SimpsonW = (3 + (-1) ** j - delt) / 3
    if S0 >= 0.95 * K:
        FFTFunc = np.exp(1j * b * vo) * modcharFunc * eta * SimpsonW
        payoff = (fft(FFTFunc)).real
        CallValueM = np.exp(-alpha * k) / np.pi * payoff
    else:
        FFTFunc = (np.exp(1j * b * vo) *
                   (modcharFunc1 - modcharFunc2) *
                   0.5 * eta * SimpsonW)
        payoff = (fft(FFTFunc)).real
        CallValueM = payoff / (np.sinh(alpha * k) * np.pi)
    pos = int((k + b) / eps)
    CallValue = CallValueM[pos] * S0
    # klist = np.exp((np.arange(0, N, 1) - 1) * eps - b) * S0
    return CallValue  # , klist[pos - 50:pos + 50]
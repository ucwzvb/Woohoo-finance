import numpy as np


def black_scholes_characteristic_function(u, x, r, sigma, mt, t):
    y = np.exp(1j * u * (x + r * (mt-t)) - (1j * u + u ** 2) * sigma ** 2 / 2 * (mt-t))
    return y




def laplace_transform_vanilla(z, strike):
    # Laplace transform of the function f(x) = max(e^x - K, 0) or f(x) = max(K - e^x, 0)
    return (strike ** (1 - z)) / (z * (z - 1))



def laplace_transform_vanilla_0(z):
    # laplace transform for vanilla option with strike=exp(kappa)
    # for for pricing via fast fourier transform
    return 1/((z-1)*z)


def fast_fourier_bs_eu(spot, strikes, r, sigma, mt, option_type="C", n=10000, m=400, t=0):
    '''# spot: underlying spot price
    # strikes: strike prices can be a list of multiple strikes
    # r: rik free interest rate
    # sigma: volatility
    # mt: time to maturity in years
    # option_type: Type of the option either call or put
    # n, m: delta=m/n is the mesh size of the integral approximation via the midpoint rule, m should be large and m/n should be small
    # t: time at which to evaluate the option

    # returns a list of three values:
    # 1. option values corresponding to the inputed strikes
    # 2. options values without interpolation
    # 3. strikes corresponding to the prices without interpolation
    # ------------------------------------------------------------------------------------------------------------------'''
    if option_type == "C":
        R = 1.1
    elif option_type == "P":
        R = -0.1
    else:
        print("ERROR, option_type must be either 'call', or 'put'")
        return

    def g(u):
        return laplace_transform_vanilla_0(R+1j*u)*black_scholes_characteristic_function(u-1j*R, np.log(spot), r, sigma, mt, t)

    delta = m/n
    kappa1 = np.log(np.min(strikes))

    x = np.zeros(n, dtype=np.complex128)
    for i in range(1, n):
        x[i-1] = g((i - 0.5)*delta) * delta * np.exp(-1j*(i-1)*delta*kappa1)

    # perform DFT using the efficient FFT algorithm
    x_hat = np.fft.fft(x)
    # compute vector kappa
    kappa_m = kappa1 + (np.arange(1, n+1) - 1) * 2 * np.pi / m

    strikes_fft = np.exp(kappa_m)
    # finally compute the option prices
    vt_fft = (np.exp(-r*(mt-t) + (1-R)*kappa_m))/np.pi * np.real(x_hat*np.exp(-1j * delta * kappa_m / 2))
    # interpolate strike values
    vt_fft_interpolated = np.interp(strikes, strikes_fft, vt_fft)
    return vt_fft_interpolated
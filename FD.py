#欧式期权有限差分法
import numpy as np

def get_matrix(M, K, delta_S,type):
    '''Set up three boundary conditions.'''
    # generate a (M+1)*(M+1) matrix
    f_matrx = np.matrix(np.array([0.0]*(M+1)*(M+1)).reshape((M+1, M+1)))
    # condition_1: when S=0，call=0
    f_matrx[:,0] = 0.0
    # condition_2：price=max(δS*j-St, 0)
    for i in range(M + 1):
        try:
            if type == 'C':
                f_matrx[M, i] = float(max(delta_S * i - K, 0))
            elif type == 'P':
                f_matrx[M, i] = float(max(K - delta_S * i, 0))
        except:
            print('please confrim all the paramnetars')
    # condition_3：S=S_max call=S_max-St
    f_matrx[:,M] = float(K)
    print("f_matrix shape : ", f_matrx.shape)
    return f_matrx

def calculate_coeff(j,r,sigma,delta_T):
    '''calculate coefficients.'''
    vj2 = (sigma * j)**2
    aj = 0.5 * delta_T * (r * j - vj2)
    bj = 1 + delta_T * (vj2 + r)
    cj = -0.5 * delta_T * (r * j + vj2)
    return aj, bj, cj

def get_coeff_matrix(M, r, sigma, delta_T):
    '''calculate coefficient matrix.

    Parameters
    ==========
    M : int
        number of grid points in space dimension
    r : float
        constant, risk-less short rate
    sigma : float
        volatility
    delta_T : float
        delta t
    Returns
    =======
    mat : matrix
        coefficient matrix
    '''
    mat = np.matrix(np.array([0.0]*(M-1)*(M-1)).reshape((M-1, M-1)))
    a1, b1, c1 = calculate_coeff(1, r, sigma, delta_T)
    am_1, bm_1, cm_1 = calculate_coeff(M - 1, r, sigma, delta_T)
    mat[0,0] = b1
    mat[0,1] = c1
    mat[M-2, M-3] = am_1
    mat[M-2, M-2] = bm_1
    for i in range(2, M-1):
        a, b, c = calculate_coeff(i,r,sigma,delta_T)
        mat[i-1, i-2] = a
        mat[i-1, i-1] = b
        mat[i-1, i] = c
    print("coeff matrix shape : ",  mat.shape)
    return mat

def FD_option_value(M,S, K, r,sigma,delta_S,delta_T,type):
    '''B-S finite difference method European option valuation.

    Parameters
    ==========
    M : int
        number of grid points in space dimension
    S : int
       stock price
    K : float
        strike price
    delta_S : float
        boundary price/M
    type : string
        call/put option
    r : float
        constant, risk-less short rate
    sigma : float
        volatility
    delta_T : float
        delta t
    Returns
    =======
    price : float
        European option present value at t
    '''
    f_matrx = get_matrix(M, K, delta_S,type)
    matrx = get_coeff_matrix(M,r,sigma,delta_T)
    inverse_m = matrx.I
    for i in range(M, 0, -1):
        # 迭代
        Fi = f_matrx[i, 1:M]
        Fi_1 = inverse_m * Fi.reshape((M-1, 1))
        Fi_1 = list(np.array(Fi_1.reshape(1, M-1))[0])
        f_matrx[i-1, 1:M]=Fi_1
    i = np.round(S/delta_S, 0)
    price = f_matrx[0, int(i)]
    return price
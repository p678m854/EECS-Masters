"""
File: dsp.py
Author: Patrick McNamee
Date: 3/18/2020

Summary:
File of personally written functions for Digital Signal Pocessing used to estimate values or derivatives. Made because I was sick of libraries using uniform interval theory so I made functions to handle nonuniform intervals.
"""


import numba
import numpy as np
import math
import copy


"""
Numerical Differentiation
"""


def get_weights(n, h):
    #Preallocations
    m = h.shape[0]
    w = np.zeros((m,1))
    w[n] = 1.0

    # A(h) * w = [0,...,0,1,0,...,0]^T
    A = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            A[i,j] = (h[j][0] ** i)/float(math.factorial(i))
    return np.matmul(np.linalg.inv(A), w)


def nth_numerical_derivative(f, x, n=1, order=1, method='central'):
    """
    Summary: Finds the nth derivative of the state vector f along the input vector x with an order of m. Does either forward,
    backwards, or central methods for the majority of the vector with backwards to the end of the vector
    
    Parameters:
        f      = history vector of f(x_i)
        x      = history vector of x_i which is strictly monotonically increasing
        n      = nth order derivative
        order  = order of accuracy, which is used to determine the number of required points
        method = {'central', 'forward', 'backward'}
    
    Results:
        dndxn = history vector of the numerical nth order
    """
    
    #Preallocations
    m = x.shape[0]
    n_pts = n + order
    
    #check to make sure enough points for order
    if m < n_pts:
        error("Not enough points!")
    
    dndxn = np.zeros((m,1))
    
    if method == 'central':
        l_offset = int((n_pts - 1)/2)
        u_offset = n_pts - 1 - l_offset
        #print("(%i, %i)" % (l_offset, u_offset))
    elif method == 'forward':
        l_offset = n_pts - 1
        u_offset = 0
    elif method == 'backward':
        l_offset = 0
        u_offset = n_pts - 1
    else:
        error("Method must be in ['central', 'forward', 'backward']")
    
    # Iterate through the history vectors
    for i in range(m):
        if i < n_pts:
            h       = x[:n_pts] - x[i]
            f_local = f[:n_pts]
        elif i < (m - n_pts):
            h       = x[(i - l_offset):(i + u_offset + 1)] - x[i]
            f_local = f[(i - l_offset):(i + u_offset + 1)]
        else:
            h       = x[-n_pts:] - x[i]
            f_local = f[-n_pts:]
        h.reshape((n_pts,1))
        w = get_weights(n, h)
        dndxn[i] = sum(w*f_local)[0]

    return dndxn


"""
Savitzky-Golaz Filter
"""


def lsee_polynomial_extrapolate(tvec, x, tpred, m):
    """
    Uses the known points to fit a least squares polynomial and then extrapolates it to a predition interval
    """
    pn = tpred.shape[0] #number of prediction points
    dmdxm = np.zeros((pn, m+1))
    
    # Get coefficients
    X = np.outer(tvec, np.ones((m+1,))) # x (m+1)
        # creating rows of [t^0, t^1, ..., t^m]
    for j in range(m+1):
        X[:,j] = X[:,j] ** j
    Xt = np.transpose(X) #(m+1) x ws
    #Least squares solution (m+1)x1
    theta = np.matmul(np.linalg.inv(np.matmul(Xt,X)),
                      np.matmul(Xt, np.transpose(x)))
    
    #For prediction interval
    Xp  = np.outer(tpred, np.ones((m+1,)))
    for j in range(m+1):
        Xp[:,j] = Xp[:,j] ** j
    #Start populating the prediction
    diff_multiplier = list(range(1,m+1))
    for i in range(m+1):
        dmdxm[:,i] = np.matmul(Xp, theta)
        if i != m:
            Xp = Xp[:,:-1]
            theta = theta[1:]*diff_multiplier
            diff_multiplier = diff_multiplier[:-1]
    
    return dmdxm


# Creating function for arbitary polynomial fitting
def central_sg_filter(tvec, xvec, m=5, window=13):
    """
    Based off the Savitzky-Golay filter which uses a least-squares estimation to determine a local polynomial. 
    
    Parameters:
        tvec   = (n,) numpy array for time vector which is not uniform
        xvec   = (n,) numpy array of the noise time
        m      = mth order polynomial
        window = number of points to be looking at. Should be odd so that the smoothing is being done 
                [-(j-1)/2,...,-1,0,1,...,(j-1)/2]
    
    Output:
        dmdxm = (n,m+1) vector of 0th to mth derivatives
        
    Reference: https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter
    """
    
    #Initialize vectors
    n     = tvec.shape[0] #Length of x vector
    dmdxm = np.zeros((n,m+1)) #n x (m+1) matrix of the ith derivative
    #print(dmdxm.size())
    hs    = int((window - 1)/2) #Half window size
    
    #Conversion between polynomial coefficients and the derivatives
    diff_multipliers = np.ones(m+1)
    mfact = math.factorial(m)
    for i in range(m+1):
        diff_multipliers[i] = math.factorial(i)
    
    #Middle sections (i.e. central estimations)
    for i in range(hs, n-hs):
        tlocal = tvec[i-hs:i+hs+1] - tvec[i]
        y      = xvec[i-hs:i+hs+1] #1 x ws
        X = np.outer(tlocal, np.ones((m+1,))) #ws x (m+1)
        # creating rows of [t^0, t^1, ..., t^m]
        for j in range(m+1):
            X[:,j] = X[:,j] ** j
        Xt = np.transpose(X) #(m+1) x ws

        try:
            #Least squares solution (m+1)x1
            XtXinv = np.linalg.inv(np.matmul(Xt,X))
            theta = np.matmul(XtXinv, np.matmul(Xt, np.transpose(y)))
        except np.linalg.LinAlgError as err:
            #exact solution
            X = X[:(m+1), :(m+2)] #Get a square matrix
            theta = np.matmul(np.linalg.inv(X), np.transpose(y[:(m+1)]))

        dmdxm[i,:] = np.transpose(theta)*diff_multipliers

    #Beginning and ending sections
    """
    # Pretty sure extrapolation with non-zero derivatives is throwing things off
    for (i0, index) in zip([hs, n-hs-1], [list(range(hs)), list(range(n-hs,n))]):
        
        t0        = tvec[i0] #zeroth time
        t_predict = tvec[index] - t0 # 1 x (window - 1)/2
        
        if i0 < n/2:
            i_train = list(range((i0+1),(i0+window+2)))
        else:
            i_train = list(range((i0-window-1),i0))
        
        t_train = tvec[i_train] - t0
        x_train = xvec[i_train]
        dmdxm[index,:] = lsee_polynomial_extrapolate(t_train, x_train, t_predict, m)
    """
    #Enforcing zero order hold for edge sections - 6/7/2020
    # Beginning
    for i in range(hs):
        dmdxm[i,:] = dmdxm[hs,:]
    for i in range(n-hs, n):
        dmdxm[i,:] = dmdxm[n - hs - 1,:]
    return dmdxm


numba.njit(parallel=True)
def central_sg_filter_parallel(tvec, xvec, m=5, window=13):
    """
    Same Savitzky-Golaz but made to run on parallel cpus
    """
    
    #Initialize vectors
    n     = tvec.shape[0] #Length of x vector
    dmdxm = np.zeros((n,m+1)) #n x (m+1) matrix of the ith derivative
    #print(dmdxm.size())
    hs    = int((window - 1)/2) #Half window size
    
    #Conversion between polynomial coefficients and the derivatives
    diff_multipliers = np.ones(m+1)
    mfact = math.factorial(m)
    for i in numba.prange(m+1):
        diff_multipliers[i] = math.factorial(i)
    
    #Middle sections (i.e. central estimations)
    for i in numba.prange(hs, n-hs):
        tlocal = tvec[i-hs:i+hs+1] - tvec[i]
        y      = xvec[i-hs:i+hs+1] #1 x ws
        X = np.outer(tlocal, np.ones((m+1,))) #ws x (m+1)
        # creating rows of [t^0, t^1, ..., t^m]
        for j in range(m+1):
            X[:,j] = X[:,j] ** j
        Xt = np.transpose(X) #(m+1) x ws

        try:
            #Least squares solution (m+1)x1
            XtXinv = np.linalg.inv(np.matmul(Xt,X))
            theta = np.matmul(XtXinv, np.matmul(Xt, np.transpose(y)))
        except np.linalg.LinAlgError as err:
            #exact solution
            X = X[:(m+1), :(m+2)] #Get a square matrix
            theta = np.matmul(np.linalg.inv(X), np.transpose(y[:(m+1)]))

        dmdxm[i,:] = np.transpose(theta)*diff_multipliers

    #Beginning and ending sections
    #Enforcing zero order hold for edge sections - 6/7/2020
    # Beginning
    for i in numba.prange(hs):
        dmdxm[i,:] = dmdxm[hs,:]
    for i in numba.prange(n-hs, n):
        dmdxm[i,:] = dmdxm[n - hs - 1,:]
    return dmdxm


"""
Fourier Analysis
"""


def FFT(tvec, X, Fs):
    """
    Description:
    Inputs:
        * tvec = 1xN time vector
        * X    = NxM matrix of time domain values
        * Fs   = Sample frequency
    Outputs:
        * fvec = 1xNf vector of frequencies
        * Xf   = NfxM matrix of frequency coefficients
    """
    #Set up for frequency vector
    N = len(tvec)
    if len(X.shape) == 1:
        M = 0
    else:
        M = X.shape[1]
    
    fvec = Fs*np.arange(-N/2,N/2)/N #Both negative and positive frequencies
    
    #Set up preallocation of results
    Nf = len(fvec)
    if M == 0:
        Xf = np.zeros((Nf,), dtype=complex)
    else:
        Xf = np.zeros((Nf,M), dtype=complex)
    
    #Imaginary unit
    z = 1j

    #Loop through time vector
    for i in range(N):
        #Iterate through frequency
        # Preallocate the sine and cosine components
        c = np.cos(2*np.pi*fvec*tvec[i]) #1xNf
        s = np.sin(2*np.pi*fvec*tvec[i]) #1xNf
        
        # Calculate additional part to it
        if M == 0:
            Xf = Xf + (c - z*s)*(X[i] + 0*z)
        else:
            Xf = Xf + np.matmul(np.transpose(c - z*s), X[i,:] + 0*z)
        #NfxM = NfxM + (1xNf)'*(1xM)

    return (fvec, Xf)


"""
Traditional filters
"""


def adaptive_low_pass(t, x, fc):
    """Adaptive timestep low pass filter
    
    Inputs:
        t = N time series vector with non-uniform differences
        x = NxM state vector
        fc = cut-off frequency
    
    Outputs:
        y = NxM smoothed state vector
    """
    # Preallocate output vector
    y = np.zeros(x.shape)
    # Cut off angular rate
    omega_c = 2.*np.pi*fc
    # Time step vector
    dt = np.zeros(t.shape)
    dt[1:] = t[1:] - t[:-1]
    # Calculate the weights
    alpha = omega_c*dt/(omega_c*dt + 1)
    # Iterate through time
    for i in range(t.shape[0]):
        if i == 0:
            # first point case
            y[0] = x[0]
        else:
            y[i] = (1 - alpha[i])*y[i-1] + alpha[i]*x[i]
    return y


def moving_average(tvec, X, window):
    """Centered moving average with a specified window size.
    
    Args:
        tvec (np.ndarray): N time array
        X (np.ndarray): NxM state array
        window (float): total span of the time window
    
    Returns:
        X_f (np.ndarray): Filtered NxM state array
    """
    
    X_f = np.zeros(X.shape)
    half_window = window/2.
    
    for i in range(tvec.shape[0]):
        t_sample = tvec[i]
        kernel_values = X[np.logical_and(
            tvec >= t_sample - half_window,
            tvec <= t_sample + half_window
        )]
        X_f[i] = np.mean(kernel_values, axis=0)
    
    return X_f


def IQR_anomoly_test(X, k=1.5):
    """Inter-quartile range filter over N data points in M dimensions.
    
    Args:
        X (np.ndarray): NxM array of real values
    Returns:
        index that is true if Q1[m] - k*IQR[m] <= X[i, m] <= Q3[m] + k*IQR[m] for all m in M
    """
    
    # Get Q2
    Q2 = np.median(X, axis=0)
    
    # Preallocate Q1 and Q3
    Q1 = np.zeros(Q2.shape)
    Q3 = np.zeros(Q2.shape)
    
    
    # Loop over dimensions
    for j in range(X.shape[1]):
        Q1[j] = np.median(X[X[:,j] < Q2[j],j])
        Q3[j] = np.median(X[X[:,j] > Q2[j],j])
        
    # Calculate IQR
    IQR = Q3 - Q1
    
    # Lower and upper bounds
    lb = Q1 - k*IQR
    ub = Q3 + k*IQR
    
    return np.all(
        np.logical_and(
            X > lb,
            X < ub
        ),
        axis=1  # All points along dimensions are within their respective bounds
    )
    
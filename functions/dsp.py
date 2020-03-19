"""
File: dsp.py
Author: Patrick McNamee
Date: 3/18/2020

Summary:
File of personally written functions for Digital Signal Pocessing used to estimate values or derivatives. Made because I was sick of libraries using uniform interval theory so I made functions to handle nonuniform intervals.
"""

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
        
    return dmdxm

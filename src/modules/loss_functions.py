"""
File: CUSTOM-LOSS-FUNCTIONS.PY
Author: Patrick McNamee
Date: 12/15/2020

Brief: Custom loss functions for tensorflow neural networks.
"""


import tensorflow as tf


@tf.function
def roughness_penalty(
    y_pred,
    degree=tf.constant(5, dtype=tf.int32),
):
    """Computes a penalty of roughness from a LSE best-fit polynomial.
    
    Description: For each sample, the cost function attempts to find the sum of squared error
        of the points from being in a smooth class $C^d$ specified by a given degree ($d$).
        Note that $\forall\ k\geq 0,\ C^{k+1} \subsetneq C^k$ and membership of $C^k$ is determined
        if the function is $k$ differentiable. Polynomials are used to determine if the points
        belong in $C^d$ as $p$ with $\deg(p)=d \in C^d$ and all the points can be estimated
        by a Taylor series expansion about $x=\vec{0}$. The development assumption was that
        all points predicted belong to a $f\in C^0$ as the problem was dealing with a continuous
        stochastic function so $\min_{p\in P^d} \sum_{i=1}^n (f(x_i) - p(x_i))^2$ would give
        a distance (pseudo-metric, dealing with element and a set) of $f$ from $C^d$.
    
    Args:
        y_pred (tf.Tensor): NxM tensor outputed by the neural network corresponding to N samples of M points.
        degree (tf.constant): degree of polynomial used to determine distance.
    
    Returns:
        sample averaged sum of squared error of predicted points from least squares polynomial fit.
    """
    # Compute X for LSE where input is simply and integer list
    base, power = tf.meshgrid(
        tf.keras.backend.cast_to_floatx(tf.range(tf.shape(y_pred)[1])),
        tf.keras.backend.cast_to_floatx(tf.range(degree + tf.constant(1))),
        indexing='ij'
    )
    X = tf.math.pow(base, power)
    # There will be an issue xith (XtX)^-1 if X is reducable to a square matrix
    XtXinvXt = tf.linalg.matmul(tf.linalg.inv(tf.linalg.matmul(X, X, transpose_a=True)), X, transpose_b=True)
    
    # Loop over given data
    def smoothness_sse(y_pred_sample):
        """Calculate SSE from polynomial"""
        # Cast y from 
        y, _ = tf.meshgrid(
            y_pred_sample,
            tf.keras.backend.cast_to_floatx(tf.range(tf.constant(1))),
            indexing='ij'
        )
        # LS estimator
        theta = tf.linalg.matmul(XtXinvXt, y)
        error_continuity = y - tf.linalg.matmul(X, theta)
        return tf.keras.backend.sum(tf.keras.backend.square(error_continuity))
    
    dist_from_Cd = tf.vectorized_map(
        fn=smoothness_sse,
        elems=y_pred
    )
    return tf.math.reduce_mean(dist_from_Cd)
    
@tf.function
def mse_and_roughness(
    y_actual, y_pred,
    degree=tf.constant(5),
    alpha=tf.constant(0.1)
):
    """Linear weighting between MSE and average smoothness penalty"""
    mse_points = tf.keras.losses.MSE(y_actual, y_pred)  # mean squared error
    mrp = roughness_penalty(y_pred, degree)  # mean smoothness penalty
    
    return (tf.constant(1.) - alpha)*mse_points + alpha*tf.keras.backend.sum(mrp)

@tf.function
def bezier_loss(y_actual, y_pred):
    """Applies a SSE error from a predicted Bezier Curve.
    
    Description: Bezier Curves are polynomial curves formed by points parametarized by t \in [0,1].
        A polynomial curve of degree D is established using D+1 points which allows for a continuous output.
        There is not a restriction on the points such that they be either unique or specifically spaced so
        the shapes can be very tailorable. This loss function takes N samples of M points and returns the N sum
        of squared errors of the M points from the D+1 output points.
    
    Args:
        y_actual: NxMx(K+1) tensor where y[:,:,0] is the t value of the interval and  y[:, :, 1:] is the output value
                  in K dimensional space. t should be in the interval [0, 1] but it is not necessary.
        y_pred: Nx(D+1)xK tensor, with same dimension allocations expect the t variable is excluded.
    
    Returns: float error representing \sum_{n\in N} \sum_{m\in M} \sum_{k=1}^{K-1} (y_{n,m,k} - b_k(t_{n,m}))^2
    """
    
    # Determine output dimension size and polynomial degree
    Dp1 = tf.shape(y_pred)[1]  # Degree plus 1
    K = tf.shape(y_pred)[2]
    
    # Preallocate factorial parts
    d = tf.range(Dp1)
    factorial = tf.math.cumprod(tf.range(tf.constant(1), Dp1 + tf.constant(1)), exclusive=True)
    nchoosek = tf.keras.backend.cast_to_floatx(factorial[Dp1 - tf.constant(1)]/(factorial*factorial[::-1]))
    
    # Get bezier coefficients for all 
    def bezier_coefficients(t):
        """Given t, generates the D+1 coefficents for the spline"""
        tk = tf.math.pow(
            tf.repeat(t, repeats=Dp1),
            tf.range(tf.keras.backend.cast_to_floatx(Dp1),dtype=tf.float32)
        )  # t^k
        Omtnmk = tf.math.pow(
            tf.repeat(tf.constant(1.) - t, repeats=Dp1),
            tf.keras.backend.cast_to_floatx(tf.range(
                start=(Dp1 - tf.constant(1)),
                limit=tf.constant(-1),
                delta=tf.constant(-1)
            ))
        )  # (1-t)^{n-k}
        # nchoosek = tf.vectorized_map(fn=mod_nchoosek, elems=d) # n choose k
        return nchoosek*tk*Omtnmk
    
    def bezier_coefficient_list(sample):
        """Find bezier coefficients for each training point of a sample."""
        bc_list = tf.vectorized_map(
            fn=bezier_coefficients,
            elems=sample[:,0]  # M list of t
        ) # Mx(D+1)
        return bc_list
    
    ts = y_actual[:,:,0]
    
    coefficients = tf.vectorized_map(
        fn=bezier_coefficient_list,
        elems=y_actual
    )  # NxMx(D+1)
    
    # With the coefficients, find the NxMxK tensor of predicted values
    predicted_points = tf.matmul(coefficients, y_pred) # a NxMx(D+1) times Nx(D+1)xK -> NxMxK tensor
    
    error = y_actual[:, :, 1:] - predicted_points
    
    # Find sum of squared l2 norm and average by N
    return tf.keras.backend.sum(tf.keras.backend.square(
        error
    ))/tf.keras.backend.cast_to_floatx(tf.shape(y_actual)[0])

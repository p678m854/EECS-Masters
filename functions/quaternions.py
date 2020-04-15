"""
File: Quaternions
Author: Patrick McNamee
Date: 4/15/2020

Description: Python file to handle quaternions and quaternion transformations
"""

import math
import numpy as np


def Rmatrix_from_quaternions(qw, qx, qy, qz):
    """From a quaternion to aircraft euler angles.
    
    Keyword arguments:
    qw -- rotational component
    qx -- x-axis component
    qy -- y-axis component
    qz -- z-axis component
    """
    q = np.array([qw, qx, qy, qz], dtype=np.float64, copy=True)
    ssq = np.dot(q,q) #Sum of squares norm
    Rmatrix = None #Rotation matrix
    
    #Machine Precission work
    if ssq < 10 ** -6:
        Rmatrix = np.identity(3)
    else:
        q *= math.sqrt(2/ssq)
        q = np.outer(q,q)
        
        """
        From:
            Representing Attitude: Euler Angles, Unit Quaternions, and Rotation Vectors
                by James Diebel, Stanford University (2006)
        Eq 125:
        | q0^2 + q1^2 - q2^2 - q3^2 | 2(q1*q2 + q3*q4)          | 2(q1*q3 - q0*q2)          |
        | 2(q1*q2 - q3*q4)          | q0^2 - q1^2 + q2^2 - q3^2 | 2(q2*q3 + q0*q1)          |
        | 2(q1*q3 + q0*q2)          | 2(q2*q3 - q0*q1)          | q0^2 - q1^2 - q2^2 + q3^2 |
        """
        
        Rmatrix = np.array((
            (1.0 - q[2,2] - q[3,3],       q[1,2] + q[0,3],       q[1,3] - q[0,2]),
            (      q[1,2] - q[0,3], 1.0 - q[1,1] - q[3,3],       q[2,3] + q[0,1]),
            (      q[1,3] + q[0,2],       q[2,3] - q[0,1], 1.0 - q[1,1] - q[2,2])
        ), dtype=np.float64)
        
    return Rmatrix
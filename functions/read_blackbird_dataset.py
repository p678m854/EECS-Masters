# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 20:34:19 2020

@author: Patrick
"""

#Import libraries
import math
import numpy as np
import pandas as pd
import io
import requests

#Self library
import dsp
import quaternions

def quaternion_to_euler(qw, qx, qy, qz):
    """From a quaternion to aircraft euler angles.
    
    Keyword arguments:
    qw -- rotational component
    qx -- x-axis component
    qy -- y-axis component
    qz -- z-axis component
    """
    q = np.array([qw, qx, qy, qz], dtype=np.float64, copy=True)
    ssq = np.dot(q,q) #Sum of squares norm
    Rmatrix = None    #Rotation matrix
    
    #Machine Precission work
    if ssq < 10 ** -6:
        Rmatrix = np.identity(3)
    else:
        q *= math.sqrt(2/ssq) #The 2 is included for ease of formulas
        q = np.outer(q,q)     #Symmetric matrix, really only need upper diagonal
        
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

    #Determine the Euler angles (takes simpliest solution)
    theta = np.arcsin(-Rmatrix[0,2])           #Assuming theta in [-pi/2, pi/2]
    phi = np.arctan(Rmatrix[1,2]/Rmatrix[2,2]) #Assuming phi in [-pi/2, pi/2]

    Rphi = np.array((
        (1,             0,             0),
        (0,  math.cos(phi), math.sin(phi)),
        (0, -math.sin(phi), math.cos(phi))
    ), dtype=np.float64)
    Rtheta = np.array((
        (math.cos(theta), 0, -math.sin(theta)),
        (              0, 1,                0),
        (math.sin(theta), 0,  math.cos(theta))
    ), dtype=np.float64)
    
    #Determine rotation matrix by psi
    Rpsi = np.matmul(np.transpose(np.matmul(Rphi,
                                            Rtheta)),
                    Rmatrix)
    
    psi = np.arctan2(Rpsi[0,1], Rpsi[0,0])
    
    #return (phi, theta, psi)
    return np.array((phi, theta, psi), dtype=np.float64)


def read_blackbird_test(maneuver, yawdirection, speed):
    """Returns a pandas dataframe of timesynched variables.
    
    Kyeword arguments:
        maneuver -- String of the maneuver being performed
        yawdirection -- String of either 'Constant' or 'Forward'
        speed -- float of maximum speed in manuever
    
    """
    #Baseline url for csv files
    urlhead = "http://blackbird-dataset.mit.edu/BlackbirdDatasetData/" + \
              maneuver + "/" + \
              "yaw" + yawdirection + "/" +\
              "maxSpeed"  + str(speed).replace('.', 'p') + "/csv/"
    
    #Get 3D Position and Attitude csv file
    url = urlhead + "blackbird_slash_state.csv"
    pos_and_orien = pd.read_csv(io.StringIO(
            requests.get(url).content.decode('utf-8')))
    pos_and_orien = pos_and_orien.drop(['header',\
                                        'stamp',\
                                        'pose',\
                                        'position',\
                                        'orientation'],\
                                        axis=1)
    #Rename variables and add units
    pos_and_orien = pos_and_orien.rename(columns={"x" : "px_[m]",\
                                                  "y" : "py_[m]",\
                                                  "z" : "pz_[m]",\
                                                  "x.1" : "qx",\
                                                  "y.1" : "qy",\
                                                  "z.1" : "qz",\
                                                  "w" : "qw"})
    # Transform quaternions into Euler angles for ease of reference
    for i in range(pos_and_orien.index[-1]):
        q0 = pos_and_orien.at[i, 'qw']
        q1 = pos_and_orien.at[i, 'qx']
        q2 = pos_and_orien.at[i, 'qy']
        q3 = pos_and_orien.at[i, 'qz']
        euler_angles = quaternion_to_euler(q0, q1, q2, q3)
        pos_and_orien.at[i,'roll_[rad]']  = euler_angles[0]
        pos_and_orien.at[i,'pitch_[rad]'] = euler_angles[1]
        pos_and_orien.at[i,'yaw_[rad]']   = euler_angles[2]

    #3D position and orientation reference
    url = urlhead + "blackbird_slash_pose_ref.csv"
    pose_ref_df = pd.read_csv(io.StringIO(
            requests.get(url).content.decode('utf-8')))
    # Drop empty columns
    pose_ref_df = pose_ref_df.drop(['header',\
                                    'stamp',\
                                    'pose',\
                                    'position',\
                                    'orientation'],\
                                   axis=1)
    
    #Rename variables and add units
    pose_ref_df = pose_ref_df.rename(columns={"x" : "pxr_[m]",\
                                              "y" : "pyr_[m]",\
                                              "z" : "pzr_[m]",\
                                              "x.1" : "qxr",\
                                              "y.1" : "qyr",\
                                              "z.1" : "qzr",\
                                              "w" : "qwr"})
    # Transform quaternions into Euler angles
    for i in range(pose_ref_df.index[-1]):
        q0 = pose_ref_df.at[i, 'qwr']
        q1 = pose_ref_df.at[i, 'qxr']
        q2 = pose_ref_df.at[i, 'qyr']
        q3 = pose_ref_df.at[i, 'qzr']
        euler_angles = quaternion_to_euler(q0, q1, q2, q3)
        pose_ref_df.at[i,'roll_ref_[rad]']  = euler_angles[0]
        pose_ref_df.at[i,'pitch_ref_[rad]'] = euler_angles[1]
        pose_ref_df.at[i,'yaw_ref_[rad]']   = euler_angles[2]

    #Getting IMU csv file
    url = urlhead + "blackbird_slash_imu.csv"
    imu_df = pd.read_csv(io.StringIO(
            requests.get(url).content.decode('utf-8')))
    # Drop empty columns
    imu_df = imu_df.drop(['header',\
                          'stamp',\
                          'orientation',\
                          'angular_velocity',\
                          'linear_acceleration'],\
                         axis=1)
    #Rename variables and add units
    imu_df = imu_df.rename(columns={"x" : "qx",\
                                    "y" : "qy",\
                                    "z" : "qz",\
                                    "w" : "qw",\
                                    "x.1" : "omegax_[dps]",\
                                    "y.1" : "omegay_[dps]",\
                                    "z.1" : "omegaz_[dps]",\
                                    "x.2" : "ax_[m/s2]",\
                                    "y.2" : "ay_[m/s2]",\
                                    "z.2" : "az_[m/s2]"})

    #Getting PWM Signals
    url = urlhead + "blackbird_slash_pwm.csv"
    pwm_df = pd.read_csv(io.StringIO(
            requests.get(url).content.decode('utf-8')))
    # Drop empty columns
    pwm_df = pwm_df.drop(['header',\
                          'stamp'],\
                         axis=1)
    #Rename variables and add units
    for i in range(pwm_df.index[-1]):
        pwms = pwm_df.at[i,'pwm'][1:-1].split(", ")
        pwm_df.at[i,'PWM1'] = int(pwms[0])
        pwm_df.at[i,'PWM2'] = int(pwms[1])
        pwm_df.at[i,'PWM3'] = int(pwms[2])
        pwm_df.at[i,'PWM4'] = int(pwms[3])
    pwm_df.drop(['pwm'], axis=1, inplace=True)
    
    #Get Motor RPM csv file
    url = urlhead + "blackbird_slash_rotor_rpm.csv"
    motors_df = pd.read_csv(io.StringIO(
            requests.get(url).content.decode('utf-8')))
    # Drop empty columns
    motors_df = motors_df.drop(['header',\
                                'stamp',\
                                'sample_stamp',\
                                '-.3',\
                                'secs.4',\
                                'nsecs.4',\
                                'rpm'],\
                               axis=1)
    #Rename variables and add units
    motors_df = motors_df.rename(columns={"-"       : "secs_m1",\
                                          "secs.1"  : "nsecs_m1",\
                                          "nsecs.1" : "secs_m2",\
                                          "-.1"     : "nsecs_m2",\
                                          "secs.2"  : "secs_m3",\
                                          "nsecs.2" : "nsecs_m3",\
                                          "-.2"     : "secs_m4",\
                                          "secs.3"  : "nsecs_m4",\
                                          "nsecs.3" : "motor_vec"})
    #Adding in individual motor rpms
    for i in range(motors_df.index[-1]):
        rpms = motors_df.at[i,'motor_vec'][1:-1].split(',')
        motors_df.at[i,'rpm1'] = float(rpms[0])
        motors_df.at[i,'rpm2'] = float(rpms[1])
        motors_df.at[i,'rpm3'] = float(rpms[2])
        motors_df.at[i,'rpm4'] = float(rpms[3])
    motors_df.drop(['motor_vec'], axis=1, inplace=True)
    #Drop motor seconds as rpm is being synced to csv file's rosbagTimestamp
    motors_df.drop(['secs_m1', 'nsecs_m1','secs_m2', 'nsecs_m2',
                    'secs_m3', 'nsecs_m3','secs_m4', 'nsecs_m4'], 
                    axis=1, inplace=True)
    
    #Merging all dataframes
    #Drop mention of seconds and nano-seconds as well as frames and sequences
    drop_columns = ['secs', 'nsecs', 'frame_id', 'seq']
    
    pos_and_orien = pos_and_orien.drop(drop_columns, axis=1)
    pose_ref_df   = pose_ref_df.drop(drop_columns, axis=1)
    imu_df        = imu_df.drop(drop_columns, axis=1)
    pwm_df        = pwm_df.drop(drop_columns, axis=1)
    motors_df     = motors_df.drop(drop_columns, axis=1)
    # Set all non motor dataframe index to the timestamp
    pos_and_orien.set_index('rosbagTimestamp', inplace=True)
    pose_ref_df.set_index('rosbagTimestamp', inplace=True)
    imu_df.set_index('rosbagTimestamp', inplace=True)
    pwm_df.set_index('rosbagTimestamp', inplace=True)
    motors_df.set_index('rosbagTimestamp', inplace=True)
    
    #Combine into overal dataframe
    test_df = pd.concat([pos_and_orien, 
                         pose_ref_df, 
                         imu_df, 
                         pwm_df, 
                         motors_df], sort=True)
    return test_df

def imu_installation_correction(test):
    """
    Input:
        * test: Blackbird dataset in pandas dataframe
    Output:
        * test: Dataset with installation correction fixed
    """
    #Accelerometer
    test.loc[:,['ax_[m/s2]', 'ay_[m/s2]']] = test.loc[:,['ay_[m/s2]', 'ax_[m/s2]']].values
    test.loc[:,['ax_[m/s2]']] = -1.*test.loc[:,['ax_[m/s2]']].values
    #Gyroscope
    test.loc[:,['omegax_[dps]', 'omegay_[dps]']] = test.loc[:,['omegax_[dps]',
                                                               'omegay_[dps]']].values
    test.loc[:,['omegax_[dps]']] = -1.*test.loc[:,['omegax_[dps]']].values
    return test

def inertial_position_derivatives_estimation(test):
    """
    Input:
        * test: Blackbird dataset in a pandas dataframe
    Output:
        * test: Dataset with added velocity/acceleration estimates using a SG filter and the quaternions
    """
    #Get columns of interest and indexes
    subset = test[['px_[m]', 'py_[m]', 'pz_[m]']].dropna()
    ind = subset.index
    
    #Time vector for SG
    rbts2s = 10 ** -9
    tvec = (subset.index - subset.index[0])*rbts2s
    tvec = tvec.astype('float')
    
    #Iterate through positions
    for axis in ['x', 'y', 'z']:
        #Get values
        p = subset[('p' + axis + '_[m]')].values
        #Do estimation
        p_est = dsp.central_sg_filter(tvec, p, m=3, window=27)
        
        #Store values in a new dataframe
        cols = [('p' + axis + '_[m]_est'),
                ('v' + axis + '_I_[m/s]'),
                ('a' + axis + '_I_[m/s2]')]
        data = {cols[0] : p_est[:,0], cols[1] : p_est[:,1], cols[2] : p_est[:,2]}
        df = pd.DataFrame(data, columns=cols)
        df.index = ind
        test = pd.concat([test, df], sort=True)
        
    return test

def consistent_quaternions(test):
    """
    Input:
        * test: Blackbird dataset in a pandas dataframe
    Output:
        * test: Dataset with added velocity/acceleration estimates using a SG filter and the quaternions
    """
    
    for cols in [['qw', 'qx', 'qy', 'qz'], ['qwr', 'qxr', 'qyr', 'qzr']]:
        #get quaternions and their indexes
        subset = test[cols].dropna()
        ind = subset.index
        q = subset.values
    
        #Find where the series reverses
        reversed_i = None
        for i in range(1,len(ind)):
            if ind[i] < ind[i-1]:
                reversed_i = i
                break
            else:
                pass
        #Removed reversed section
        if reversed_i != None:
            ind = ind[:reversed_i]
            q = q[:reversed_i]
            
        #Consistency check
        for i in range(1,len(q)):
            q[i] = quaternions.quaternion_solution_check(q[i-1], q[i])
        
        #Make new
        df = pd.DataFrame(q, columns=cols, index = ind)
        test = test.drop(cols, axis=1)
        test = pd.concat([test, df], sort=True)
    
    return test
    
def inertial_quaternion_derivatives_estimation(test):
    """
    Input:
        * test: Blackbird dataset in a pandas dataframe
    Output:
        * test: Dataset with added quaternion angular velocity/acceleration estimates using a SG filter and the quaternions
    """
    #Get columns of interest and indexes
    subset = test[['qw', 'qx', 'qy', 'qz']].dropna()
    ind = subset.index
    
    #Time vector for SG
    rbts2s = 10 ** -9
    tvec = (subset.index - subset.index[0])*rbts2s
    tvec = tvec.astype('float')
    
    #Iterate through positions
    for axis in ['w', 'x', 'y', 'z']:
        #Get values
        q = subset[('q' + axis)].values
        #Do estimation
        q_est = dsp.central_sg_filter(tvec, q, m=5, window=27)
        
        #Store values in a new dataframe
        cols = [('q' + axis + '_est'),
                ('qdot' + axis),
                ('qdotdot' + axis)]
        data = {cols[0] : q_est[:,0], cols[1] : q_est[:,1], cols[2] : q_est[:,2]}
        df = pd.DataFrame(data, columns=cols, index=ind)
        test = pd.concat([test, df], sort=True)
        
    return test

def body_angular_derivative_estimate(test):
    """
    Input:
        * test: Blackbird dataset in a pandas dataframe with the first and second time derivative of the quaternion vector.
    Output:
        * test: Dataset with added angular velocity/acceleration estimates from quaternions. There are two solutions available but the recorded one will be the solution that minimizes the estimated average squared L2 between the quaternion omega vector and the onboard IMU vector.
    """
    #Constants
    rbts2s = 10 ** -9
    
    #Getting quaternions, 1st time derivative, and time vector
    q = test[['qw_est', 'qx_est', 'qy_est', 'qz_est']].dropna()
    t_q = q.index
    t_q = (t_q - test.index[0]) * rbts2s
    t_q = t_q.astype('float')
    q = q.values
    qdot = test[['qdotw', 'qdotx', 'qdoty', 'qdotz']].dropna().values
    
    #Getting IMU values and time derivative
    omega_imu = test[['omegax_[dps]', 
                      'omegay_[dps]', 
                      'omegaz_[dps]']].dropna()
    t_imu = omega_imu.index
    t_imu = (t_imu - test.index[0]) * rbts2s
    t_imu = t_imu.astype('float')
    omega_imu = omega_imu.values

    #Constuct the two omega vectors from the quaternions
    omega_q  = np.zeros((len(t_q),3))
    omega_qa = np.zeros((len(t_q),3))
    
    for i in range(len(t_q)):
        omega_q[i]  = quaternions.Omega_from_quaternions(q[i], qdot[i])
        omega_qa[i] = quaternions.Omega_from_quaternions_alt(q[i], qdot[i])
        
    #Iterate through IMU series determining errors
    N = len(t_imu)
    M = len(t_q)
    e1 = 0.
    e2 = 0.
    qi = 0
    for i in range(N):
        t = t_imu[i]
        while t_q[qi] < t and qi < M:
            qi += 1
        #End conditions of quaternion index
        if qi == M:
            break #
        elif qi == 0:
            pass #Quaternions started after IMU is on
        else:
            #Linear interpolate
            t1 = t_q[qi-1]
            t2 = t_q[qi]
            o1 = (t - t2)/(t1 - t2)*omega_q[qi - 1] + (t - t1)/(t2 - t1)*omega_q[qi]
            o2 = (t - t2)/(t1 - t2)*omega_qa[qi - 1] + (t - t1)/(t2 - t1)*omega_qa[qi]
            #Calculate errors
            e1 += (((o1[0] - omega_imu[i,0]) ** 2) + 
                   ((o1[1] - omega_imu[i,1]) ** 2) + 
                   ((o1[2] - omega_imu[i,2]) ** 2))/N
            e2 += (((o2[0] - omega_imu[i,0]) ** 2) + 
                   ((o2[1] - omega_imu[i,1]) ** 2) + 
                   ((o2[2] - omega_imu[i,2]) ** 2))/N
    
    # Determine best solution
    sol = None
    if e1 <= e2:
        sol = 1
    else:
        sol = 2
        omega_q = omega_qa
        
    #Get omdega dot estimates
    omegadot = np.zeros((M,3))
    for i in range(M):
        if sol == 1:
            omegadot[i] = quaternions.omegadot_from_quaternions(
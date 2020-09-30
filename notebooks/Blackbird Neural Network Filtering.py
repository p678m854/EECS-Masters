#!/usr/bin/env python
# coding: utf-8

# # Blackbird Neural Network Filtering
# 
# This notebook explores the use of neural networks to act as an onboard an realtime fast filter instead of the slower Savitzky-Golaz filter currently being used. This would allow for a network to give reasonably accurate state information for some arbitrary controller. It may be faster than current methods.
# 
# ## Load in Data
# 
# First thing is that the library and flight test data need to be imported before any training can be done.

# In[1]:


import importlib
import os
import sys
import time
sys.path.append(os.path.abspath('../functions'))

import read_blackbird_dataset as rbd
import dsp
import quaternions

#get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numba
import numpy as np
import pandas as pd

from sklearn import model_selection
import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots


# In[2]:


# Initial read in
print('Time elapsed breakdown')

t0 = time.time()
test_df = rbd.read_blackbird_test('figure8', 'Constant', 0.5)
t1 = time.time()
print("\tInitial Flight Data Read in = %f [s]" % (t1 - t0))

rbd.imu_installation_correction(test_df)
t2 = time.time()
print('\tIMU correction = %f [s]' % (t2 - t1))

test_df = rbd.inertial_position_derivatives_estimation(test_df)
t3 = time.time()
print('\tInertial position derivative estimates = %f [s]' % (t3 - t2))

test_df = rbd.gyroscope_derivatives_estimation(test_df)
t4 = time.time()
print('\tGyroscope derivative estimates = %f [s]' % (t4 - t3))

test_df = rbd.consistent_quaternions(test_df)
t5 = time.time()
print('\tConsistent quaternions = %f [s]' % (t5 - t4))

test_df = rbd.inertial_quaternion_derivatives_estimation(test_df)
t6 = time.time()
print('\tQuaternion derivate estimates = %f [s]' % (t6 - t5))

test_df = rbd.body_quaternion_angular_derivative_estimate(test_df)
t7 = time.time()
print('\tBody rates from quaternion derivatives = %f [s]' % (t7-t6))

test_df = rbd.motor_scaling(test_df)
t8 = time.time()
print('\tRescale motor angular rates = %f [s]' % (t8 - t7))

test_df = rbd.motor_rates(test_df)
t9 = time.time()
print('\tMotor derivates = %f [s]' % (t9 - t8))

test_df = rbd.quaternion_body_acceleration(test_df)
t10 = time.time()
print('\tPut in body frame accelerations = %f' % (t10 - t9))

test_df = rbd.on_ground(test_df)
t11 = time.time()
print('\tBoolean for flying = %f [s]' % (t11 - t10))

print('Total time elapsed = %f' % (t11 - t0))
test_df.info()


# ## Baseline TensorFlow
# 
# First model to establish a baseline of Tensorflow and Keras is to estimate the acceleration due to gravity in the quadcopter body frame. This is convenient since this acceleration is for the most part time invariant and the time series of quaternions and the calculated acceleration due to gravity is the same.

# In[3]:


# Isolate Input and Output from flight test
qvec = test_df[['qw', 'qx', 'qy', 'qz']].dropna().values
gvec = test_df[['ax_g|B_[m/s2]', 'ay_g|B_[m/s2]', 'az_g|B_[m/s2]']].dropna().values

# Scaling factor to g's from [-1, 1] which represents the range of the Xsens MTi-3 IMU's accelerometer
gsf = 16*9.81  # +/- 16 g range
gvec = gvec/ gsf


# In[4]:


# Split up into test and train sets (70-30 split)
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    qvec, gvec, test_size=0.30, random_state=42
)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[ ]:


# Creating a model
inputs = tf.keras.Input(shape=(4,))
x = tf.keras.layers.Dense(3, activation=tf.nn.tanh)(inputs)
x = tf.keras.layers.Dense(2, activation=tf.nn.tanh)(x)
outputs = tf.keras.layers.Dense(3, activation=tf.nn.tanh)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

"""
model.compile(
    optimizer='rmsprop',
    loss=tf.keras.losses.mse,
    metrics=tf.keras.metrics.mse,
    loss_weights=None,
    weighted_metrics=None,
    run_eagerly=None,
)
model.summary()
"""


# In[ ]:


model.predict(X_train[:10])


# In[ ]:


EPOCHS = 1000

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    verbose=0,
    validation_data=(X_test, y_test),
    callbacks=[tfdocs.modeling.EpochDots()]
)


# In[ ]:


hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


# In[ ]:


plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)


# In[ ]:


plotter.plot({'Basic': history}, metric = "mean_squared_error")
#plt.ylim([0, 10])
plt.ylabel('MAE [CosineSimilarity]')


# In[ ]:


inputs = tf.keras.Input(shape=(4,))
x = tf.keras.layers.Dense(3, activation=tf.nn.tanh)(inputs)
x = tf.keras.layers.Dense(2, activation=tf.nn.tanh)(x)
outputs = tf.keras.layers.Dense(3, activation=tf.nn.tanh)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer='rmsprop',
    loss=tf.keras.losses.mse,
    metrics=tf.keras.metrics.mse,
    loss_weights=None,
    weighted_metrics=None,
    run_eagerly=None,
)
model.summary()


# In[ ]:


# The patience parameter is the amount of epochs to check for improvement
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

early_history = model.fit(
    X_train, y_train, 
    epochs=EPOCHS,
    validation_data = (X_test, y_test),
    verbose=0, 
    callbacks=[early_stop, tfdocs.modeling.EpochDots()]
)


# In[ ]:


plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
plotter.plot({'Early': early_history}, metric = "mean_squared_error")
plt.ylabel('MAE [CosineSimilarity]')


# In[ ]:


# Showing the acceration over flight test
tvec = test_df[['qw']].dropna().index.values
tvec = (tvec - tvec[0])*(10 ** -9)
tvec = tvec.astype('float')

gvec = test_df[['ax_g|B_[m/s2]', 'ay_g|B_[m/s2]', 'az_g|B_[m/s2]']].dropna().values
gpred = model.predict(qvec)*gsf

fig, ax = plt.subplots(3,1)

ax[0].plot(tvec, gvec[:, 0], label='Known')
ax[0].plot(tvec, gpred[:, 0], label='NN estimated')
ax[1].plot(tvec, gvec[:, 1])
ax[1].plot(tvec, gpred[:, 1])
ax[2].plot(tvec, gvec[:, 2])
ax[2].plot(tvec, gpred[:, 2])

ax[0].set_ylabel('$a_x$ [m/s$^2$]')
ax[1].set_ylabel('$a_y$ [m/s$^2$]')
ax[2].set_ylabel('$a_z$ [m/s$^2$]')
ax[2].set_xlabel('Time ($t$) [s]')

plt.show()


# ## Derivative Estimation Model
# 
# This neural network model is to try and estimate the various derivatives from not readily obtainable from sensors. List of sensors we have are below:
# 
# |Sensor | Variable | Frame |
# |---:| :---:|:---:|
# |Accelerometer|$\dfrac{d^2}{dt^2}\vec{p}$|Relative|
# |Gyroscope|$\dfrac{d}{dt}\vec{\theta}$|Relative|
# |VICON|$\vec{p}$ and $\vec{\theta}$|Inertial|
# 
# Ignoring transforms between the relative and inerital frames and the noise in the measurements, the sets of variables not here are the velocity $\dfrac{d}{dt}\vec{p}$ and rotational acceleration $\dfrac{d^2}{dt^2} \theta$ which are important for the trajectory planning/control algorithms/schemes.
# 
# ### Velocity Estimation
# 
# #### Basic Neural Network
# 
# Simpliest esimation is to use a past history of $n$ relative positions so that $\vec{p}[n] = \vec{0}$ and the network is trying to predict $\dfrac{d\vec{p}}{dt}[n]$. An option input to the network is the $n$ relative sample times and this way it is possible to try to test the effects of downsampling so that the network is decently robust and more widely applicable to other vehicle platforms. Scaling the inputs and outputs will be kind of arbitrary. What appears to make the most sense is to set a maximum velocity $V_{\max}$, $n$ samples, and a nominal sampling time $\Delta t$. This form a maximum scalar position bound $p_{\max} = n\Delta t V_\max$ to scale the relative positions. There will be two strategies tested, whether a 1D estimator repeated 3 times is sufficient or if a 3D estimator is better.

# In[ ]:


# Training and testing a regular 1D
n = 10  # generate n samples for now

# get position values
p_measured = test_df[['px_[m]', 'py_[m]', 'py_[m]']].dropna().values

# Number of points and spacial dimensions
Np = p_measured.shape[0]
Nd = p_measured.shape[1]

X = np.zeros((Nd*(Np - n + 1), n))

# Iterate through dimensions
for i in range(3):
    istart = i*(Np - n)
    istop = istart + Np - n +1
    # Iterate through n points
    for j in range(1, n):
        # I know the p[n] is always zeros but I could poentially use the smoothed value instead
        dp = p_measured[n-j-1:Np-j, i] - p_measured[n-1:, i]
        X[istart:istop, j] = dp

print(X.shape)
print(X[:5, :])


# In[ ]:


# Velocoity estimates
v_est = test_df[['vx_I_[m/s]', 'vy_I_[m/s]', 'vz_I_[m/s]']].dropna().values[n-1:]
y_est = v_est.flatten('F')


# In[ ]:


import copy

# Setting vehicle parameter
dt = 1./200.  # Normal 200 Hz update rate
Vmax = 5  # m/s

# Getting a time vector for plotting
tvec = test_df[['qw']].dropna().index.values
tvec = (tvec - tvec[0])*(10 ** -9)
tvec = tvec.astype('float')
tvec = tvec[n-1:]  # Trim off unused front

fig, ax = plt.subplots(3,1)

# plot actual values
ax[0].plot(tvec, v_est[:, 0], label='Estimated')
ax[1].plot(tvec, v_est[:, 1])
ax[2].plot(tvec, v_est[:, 2])

early_stop = tf.keras.callbacks.EarlyStopping(monitor='mse')
training_histories = []

for ni in range(2, 11):
    inputs = tf.keras.Input(shape=(ni,))
    x = tf.keras.layers.Dense(max(2, int(ni/2)), activation=tf.nn.relu)(inputs)
    x = tf.keras.layers.Dense(max(2, int(ni/4)), activation=tf.nn.relu)(x)
    outputs = tf.keras.layers.Dense(1, activation=tf.nn.tanh)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer='rmsprop',
        loss=['mse'],
        metrics=['mse'],
        loss_weights=None,
        weighted_metrics=None,
        run_eagerly=None,
    )
    
    y_normed = y_est/Vmax
    X_normed = X[:, :ni]/(Vmax*n*dt)
    
    tf.random.set_seed(42)  # Meaning of life
    hist = model.fit(
        x=copy.deepcopy(X_normed), y=y_normed,
        batch_size=1000,
        epochs=100,
        verbose=0,
        callbacks=[early_stop],
        validation_split=0.3,  # 70-30 train validation split
        shuffle=True,
        validation_freq=1,
    )
    training_histories.append(hist)
    
    y_pred = model.predict(X_normed).reshape((Np-n+1, 3), order='F')
    v_pred = y_pred*Vmax
    
    ax[0].plot(tvec, v_pred[:, 0], label=("%i points" % ni), alpha=0.6)
    ax[1].plot(tvec, v_pred[:, 1], alpha=0.6)
    ax[2].plot(tvec, v_pred[:, 2], alpha=0.6)
    
ax[0].legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize='xx-small', handlelength=1)
plt.show()


# In[ ]:


# plot training history
fig, ax_t = plt.subplots(1, 1)

for line, hist in zip(ax[0].get_lines()[1:], training_histories):
    c = line.get_color()
    # Vectors for plotting
    lvec = np.array(hist.history['loss'])
    vlvec = np.array(hist.history['val_loss'])
    epoch_vec = np.array(list(range(lvec.shape[0])))
    
    ax_t.plot(epoch_vec, lvec, color=c, label=(line.get_label() + ' training loss'))
    ax_t.plot(epoch_vec, vlvec, color=c, linestyle=':', label=(line.get_label() + ' validation loss'))

fig.suptitle('Training Losses')
ax_t.legend(fontsize='xx-small', handlelength=1)
plt.show()


# In[ ]:


# Training and testing a regular 1D
n = 10  # generate n samples for now

# get position values
p_measured = test_df[['px_[m]', 'py_[m]', 'py_[m]']].dropna().values

# Number of points and spacial dimensions
Np = p_measured.shape[0]
Nd = p_measured.shape[1]

X = np.zeros((Np - n + 1, Nd, n))

# Iterate through dimensions
for i in range(3):
    istart = i*(Np - n)
    istop = istart + Np - n +1
    # Iterate through n points
    for j in range(1, n):
        # I know the p[n] is always zeros but I could poentially use the smoothed value instead
        dp = p_measured[n-j-1:Np-j, i] - p_measured[n-1:, i]
        X[:, i, j] = dp

print(X.shape)
print(X[:2, :, :])


# In[ ]:


# 3D Velocoity estimates
v_est = test_df[['vx_I_[m/s]', 'vy_I_[m/s]', 'vz_I_[m/s]']].dropna().values[n-1:]
y_est = v_est


# In[ ]:


import copy

# Setting vehicle parameter
dt = 1./200.  # Normal 200 Hz update rate
Vmax = 5  # m/s

# Getting a time vector for plotting
tvec = test_df[['qw']].dropna().index.values
tvec = (tvec - tvec[0])*(10 ** -9)
tvec = tvec.astype('float')
tvec = tvec[n-1:]  # Trim off unused front

fig, ax = plt.subplots(3,1)

# plot actual values
ax[0].plot(tvec, v_est[:, 0], label='Estimated')
ax[1].plot(tvec, v_est[:, 1])
ax[2].plot(tvec, v_est[:, 2])

early_stop = tf.keras.callbacks.EarlyStopping(monitor='mse')
training_histories = []

for ni in range(2, 11):
    y_normed = y_est/Vmax
    X_normed = X[:, :, :ni]/(Vmax*n*dt)
    
    """
    inputs = tf.keras.Input(shape=(Nd,ni))
    x = tf.keras.layers.Dense(max(2, int(ni/2)), activation=tf.nn.relu)(inputs)
    x = tf.keras.layers.Dense(max(2, int(ni/4)), activation=tf.nn.relu)(x)
    outputs = tf.keras.layers.Dense(Nd, activation=tf.nn.tanh)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    """
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=X_normed.shape[1:]))
    model.add(tf.keras.layers.Dense(max(2, int(ni/2)), activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(max(2, int(ni/4)), activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(Nd, activation=tf.nn.tanh))
    
    model.compile(
        optimizer='rmsprop',
        loss=['mse'],
        metrics=['mse'],
        loss_weights=None,
        weighted_metrics=None,
        run_eagerly=None,
    )
    
    tf.random.set_seed(42)  # Meaning of life
    hist = model.fit(
        x=copy.deepcopy(X_normed), y=y_normed,
        batch_size=1000,
        epochs=100,
        verbose=0,
        callbacks=[early_stop],
        validation_split=0.3,  # 70-30 train validation split
        shuffle=True,
        validation_freq=1,
    )
    training_histories.append(hist)
    
    y_pred = model.predict(X_normed).reshape((Np-n+1, 3), order='F')
    v_pred = y_pred*Vmax
    
    ax[0].plot(tvec, v_pred[:, 0], label=("%i points" % ni), alpha=0.6)
    ax[1].plot(tvec, v_pred[:, 1], alpha=0.6)
    ax[2].plot(tvec, v_pred[:, 2], alpha=0.6)
    
ax[0].legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize='xx-small', handlelength=1)
plt.show()


# In[ ]:


# plot training history
fig, ax_t = plt.subplots(1, 1)

for line, hist in zip(ax[0].get_lines()[1:], training_histories):
    c = line.get_color()
    # Vectors for plotting
    lvec = np.array(hist.history['loss'])
    vlvec = np.array(hist.history['val_loss'])
    epoch_vec = np.array(list(range(lvec.shape[0])))
    
    ax_t.plot(epoch_vec, lvec, color=c, label=(line.get_label() + ' training loss'))
    ax_t.plot(epoch_vec, vlvec, color=c, linestyle=':', label=(line.get_label() + ' validation loss'))

fig.suptitle('Training Losses')
ax_t.legend(fontsize='xx-small', handlelength=1)
plt.show()


# It appears that dealing with vectors has better results for estimating derivatives although vectors generally need more epochs for training by about 3 times as much. This is likely that 1 dimensional estimates had 3 times as much data as the 3D vector case.
# 
# ### 

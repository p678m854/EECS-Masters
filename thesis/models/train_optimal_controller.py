"""
File: TRAIN_OPTIMAL_CONTROLLER.PY
Author: Patrick McNamee
Date: Feb 3, 2021

Brief: Python file for training neural networks as optimal controllers.
"""


# Import appropriate python libraries
import numpy as np
import os
import sys
import tensorflow as tf
import time

# Additional imports
if __name__=="__main__":
    from thesis.data import blackbird_dataset as rbd
    from thesis.modules import loss_functions as clf
else:
    from ..data import blackbird_dataset as rbd
    from ..modules import loss_functions as clf


# File defaults
DEFAULT_TEST = ('figure8', 'Constant', 0.5)  # Default flight test for models.
DEFAULT_DOWNSAMPLING_DICT = {  # Downsampling of values to ~10 Hz for 1 second windows.
    'stride_pos': 36,  # Input variable, (3 channels)
    'stride_att': 36,  # Input variable, (3 channels)
    'stride_pos_ref': 19,  # Input variable, (3 channels)
    'stride_att_ref': 19,  # Input variable, (3 channels)
    'stride_motor_speeds': 19,  # Input variable, (4 channels)
    'stride_accel': 10,  # Input variable, (3 channels)
    'stride_gyro': 10,  # Input variable, (3 channels)
    'stride_pwms': 19  # Output variable, (4 channels)
}


def tensorflow_demo_test():
    """Testing if tensorflow works using the beginner quickstart"""

    # Load MNIST dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Create the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    # Output the first prediction
    predictions = model(x_train[:1]).numpy()
    print(predictions)

    # Loss function stuff
    print(tf.nn.softmax(predictions).numpy())
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    print(loss_fn(y_train[:1], predictions).numpy())

    # Compile model
    model.compile(
        optimizer='adam',
        loss=loss_fn,
        metrics=['accuracy']
    )

    # Model training
    model.fit(x_train, y_train, epochs=5)

    # Model evaluation
    model.evaluate(x_test, y_test, verbose=2)

    # Changing model into probability
    probability_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])

    # Output the probability
    print(probability_model(x_test[:5]))


def creating_training_data_stress_test():
    """Function for running a stress test for training a neural network using TF 2.x

    Args:
        n (int): Flight test index in blackbird_dataset list.
    """

    # Script part for stress testing downloading dataset
    argc = len(sys.argv)
    print(sys.argv)
    if argc > 2:
        raise IOError(
            "Script is for stress testing only. Expected argc=1 or 2, recieved argc = %i." % argc
        )
    
    # Get user selected flight test
    n = int(sys.argv[1]) if argc == 2 else rbd.TEST_FLIGHT_LIST.index(rbd.DEFAULT_TEST_FLIGHT)

    # Bounds on accessing flight tests
    MIN_N = 0  # Minimum accessible index
    MAX_N = len(rbd.TEST_FLIGHT_LIST)  # Exclusive maximum index
    
    # Check bounds
    if n < MIN_N or n >= MAX_N:
        raise ValueError(
            "Selected flight test must be in interval [%i, %i), recieved %i." % (
                MIN_N, MAX_N, n
                ))

    # Downsampling for stress test setup (~10 Hz for all variables)
    ds_dict = {
        'stride_pos': 36,
        'stride_att': 36,
        'stride_pos_ref': 19,
        'stride_att_ref': 19,
        'stride_motor_speeds': 19,
        'stride_accel': 10,
        'stride_gyro': 10,
        'stride_pwms': 19
    }

    # Get the flight test dataframe
    t1 = time.time()
    print("Flight test: (%s, %s, %f)" % rbd.TEST_FLIGHT_LIST[n])
    print("Start loading dataset")
    test_df = rbd.cleaned_blackbird_test(*(rbd.TEST_FLIGHT_LIST[n]))
    print("Finished loading dataset")

    # Generate training data
    t2 = time.time()
    print("Started making training data")
    X, Y, tvec_y, info = rbd.generate_opt_control_test_data(
        test_df=test_df,
        past_delta_t=1,
        future_delta_t=1,
        downsample_dict=ds_dict
    )
    print("Finished making training data")

    # Free up unnecessary dataframe
    t3 = time.time()
    del test_df

    # Give some useful information
    print("Time to load dataframe: %f [s]" % (t2 - t1))
    print("Time to generate dataset: %f [s]" % (t3 - t2))
    print("Training examples: %i" % X.shape[0])
    print("Input dimensions: %i" % X.shape[1])
    print("Output dimensions: %i" % Y.shape[1])


def create_sequential_model(input_dim: int, output_dim: int, depth: int, layer_width=None, act_fun=None):
    """Create a simple sequential neural network.

    Description:
        Creates a simple tensorflow sequential neural network with specifiable widths. All activation functions are simply ReLU.

    Args:
        input_dim (int): Input dimension i.e X.shape[1] where X is a numpy array and len(X.shape) == 2.
        output_dim (int): Output dimension i.e. Y.shape[1] where Y is a numpy array and len(Y.shape) == 2
        depth (int): Number of layers in the neural network.
        layer_width (tuple or int): Specifying individual layer widths. If None, all layers are taken to be the output dimension. If an integer, then all layers have that width. If a tuple, then each element is a layer width with the assertion that len(layer_width) == depth.
        act_fun (callable or tuple(callables)): Activation function to use on layers. Sample logic as layer_width but this time with an activation function. Default is the tensorflow relu function.

    Returns:
        simple_model (tf.keras.Model): Simple sequenctial neural network model.
    """

    # Immediate assertions
    assert input_dim >= 0
    assert output_dim >= 0
    assert depth > 0

    # Do the layer width logic
    if layer_width is None:
        layer_width = (output_dim,) * depth  # Simply match output width
    elif type(layer_width) == int:
        assert layer_width > 0
        layer_width = (layer_width,) * (depth - 1) + (output_dim,)
    elif type(layer_width) == tuple:
        assert len(layer_width) == depth
        for lw in layer_width:
            assert type(lw) == int
            assert lw > 0
        assert layer_width[-1] == ouput_dim
    else:
        raise IOError(
            "layer_width needs to be None, tuple, or int rather than %s" % type(layer_width)
        )

    # Do the activation function logic
    if act_fun is None:
        act_fun = (tf.nn.relu,) * depth  # default 
    elif type(act_fun) == tuple:
        assert len(act_fun) == depth
        for af in act_fun:
            try:
                tf.keras.activations.serialize(af)
            except ValueError as ve:
                raise ValueError("One of the activation functions is invalid in tuple")
    else:
        tf.keras.activations.serialize(act_fun)
        act_fun = (act_fun,) * depth  # Match number of layers

    # Create model
    simple_model = tf.keras.Sequential()
    
    for i in range(depth):
        # For all every other layer
        if i > 0:
            simple_model.add(tf.keras.layers.Dense(
                units=layer_width[i],
                activation=act_fun[i]
            ))
        else:
            simple_model.add(tf.keras.layers.Dense(
                units=layer_width[i],
                activation=act_fun[i],
                input_shape=(input_dim,),
            ))

    return simple_model


def create_sequential_dropout_model(input_dim: int, output_dim: int, depth: int, layer_width=None, act_fun=None, dropout_rate=None):
    """Create a simple sequential neural network with dropouts after all hidden layers.

    Description:
        Creates a simple tensorflow sequential neural network with specifiable widths. All activation functions are simply ReLU if not secified.

    Args:
        input_dim (int): Input dimension i.e X.shape[1] where X is a numpy array and len(X.shape) == 2.
        output_dim (int): Output dimension i.e. Y.shape[1] where Y is a numpy array and len(Y.shape) == 2
        depth (int): Number of layers in the neural network.
        layer_width (tuple or int): Specifying individual layer widths. If None, all layers are taken to be the output dimension. If an integer, then all layers have that width. If a tuple, then each element is a layer width with the assertion that len(layer_width) == depth.
        act_fun (callable or tuple(callables)): Activation function to use on layers. Sample logic as layer_width but this time with an activation function. Default is the tensorflow relu function.
        dropout_rate (tuple or float): rate of dropouts after each hidden layers. The length of the tuple should be 2 less than the depth or 0, which ever is the maximum.

    Returns:
        simple_dropout_model (tf.keras.Model): Simple sequenctial neural network model with dropouts.
    """

    # Immediate assertions
    assert input_dim >= 0
    assert output_dim >= 0
    assert depth > 0

    # Do the layer width logic
    if layer_width is None:
        layer_width = (output_dim,) * depth  # Simply match output width
    elif type(layer_width) == int:
        assert layer_width > 0
        layer_width = (layer_width,) * (depth - 1) + (output_dim,)
    elif type(layer_width) == tuple:
        assert len(layer_width) == depth
        for lw in layer_width:
            assert type(lw) == int
            assert lw > 0
        assert layer_width[-1] == ouput_dim
    else:
        raise IOError(
            "layer_width needs to be None, tuple, or int rather than %s" % type(layer_width)
        )

    # Do the activation function logic
    if act_fun is None:
        act_fun = (tf.nn.relu,) * depth  # default 
    elif type(act_fun) == tuple:
        assert len(act_fun) == depth
        for af in act_fun:
            try:
                tf.keras.activations.serialize(af)
            except ValueError as ve:
                raise ValueError("One of the activation functions is invalid in tuple")
    else:
        tf.keras.activations.serialize(act_fun)
        act_fun = (act_fun,) * depth  # Match number of layers

    # Do the dropout logic
    if dropout_rate is None:
        dropout_rate = (0.,) * (depth - 2)  # default 
    elif type(dropout_rate) == tuple:
        assert len(dropout_rate) == min((depth - 2), 0)
        for dr in dropout_rate:
            assert type(dr) is int or type(dr) is float, "Droprate is not integer or float data type"
            if dr < 0. or dr > 1:
                raise ValueError("One of the dropout rates is invalid in tuple")
    else:
        assert type(dropout_rate) is int or type(dropout_rate) is float, "Dropout rate is not integer or float data type"
        assert dropout_rate >= 0 and dropout_rate <= 1, "Dropout rate needs to be in [0, 1]"
        dropout_rate = (dropout_rate,) * (depth - 2)  # Match number of layers

    # Pad dropout_rate to match depth
    dropout_rate = (None,) + dropout_rate + (None,)

    # Create model
    simple_model = tf.keras.Sequential()
    
    for i in range(depth):
        # For all every other layer add in specified width
        if i > 0:
            simple_model.add(tf.keras.layers.Dense(
                units=layer_width[i],
                activation=act_fun[i]
            ))
        # Input layer
        else:
            simple_model.add(tf.keras.layers.Dense(
                units=layer_width[i],
                activation=act_fun[i],
                input_shape=(input_dim,),
            ))

        # Add in necessary dropout layer
        if dropout_rate[i]:
            simple_model.add(tf.keras.layers.Dropout(
                rate=dropout_rate[i]
            ))
            pass

    return simple_model


def create_sequential_bezier_model(
        input_dim: int, output_dim: int,
        degree: int,
        depth: int,
        layer_width=None, act_fun=None
):
    """Create a simple sequential neural network with outpuing bezier splines.

    Description:
        Creates a simple tensorflow sequential neural network with specifiable widths. All activation functions are simply ReLU by default. Output is a Bezier curve.

    Args:
        input_dim (int): Input dimension i.e X.shape[1] where X is a numpy array and len(X.shape) == 2.
        output_dim (int): Output dimension i.e. Y.shape[2] where Y is a numpy array and
            len(Y.shape) == 3
        degree (int): degree of the Bezier polynomial
        depth (int): Number of layers in the neural network.
        layer_width (tuple or int): Specifying individual layer widths. If None, all layers are taken
            to be the output dimension. If an integer, then all layers have that width. If a tuple,
            then each element is a layer width with the assertion that len(layer_width) == depth.
        act_fun (callable or tuple(callables)): Activation function to use on layers. Sample logic as
            layer_width but this time with an activation function. Default is the tensorflow relu
            function.

    Returns:
        bezier_model (tf.keras.Model): Sequenctial bezier neural network model. Outputs in
            R^{(degree + 1) times output_dim}
    """

    # Immediate assertions
    assert input_dim >= 0
    assert output_dim >= 0
    assert depth > 0
    assert degree >= 0

    # Do the layer width logic
    if layer_width is None:
        layer_width = (output_dim*(degree + 1),) * depth  # Simply match output width
    elif type(layer_width) == int:
        assert layer_width > 0
        layer_width = (layer_width,) * (depth - 1) + (output_dim*(degree + 1),)
    elif type(layer_width) == tuple:
        assert len(layer_width) == depth
        for lw in layer_width:
            assert type(lw) == int
            assert lw > 0
        assert layer_width[-1] == ouput_dim*(degree + 1)
    else:
        raise IOError(
            "layer_width needs to be None, tuple, or int rather than %s" % type(layer_width)
        )

    # Do the activation function logic
    if act_fun is None:
        act_fun = (tf.nn.relu,) * depth  # default 
    elif type(act_fun) == tuple:
        assert len(act_fun) == depth
        for af in act_fun:
            try:
                tf.keras.activations.serialize(af)
            except ValueError as ve:
                raise ValueError("One of the activation functions is invalid in tuple")
    else:
        tf.keras.activations.serialize(act_fun)
        act_fun = (act_fun,) * depth  # Match number of layers

    # Create model
    bezier_model = tf.keras.Sequential()
    
    for i in range(depth):
        # For all every other layer
        if i > 0:
            bezier_model.add(tf.keras.layers.Dense(
                units=layer_width[i],
                activation=act_fun[i]
            ))
        # Input layer
        else:
            bezier_model.add(tf.keras.layers.Dense(
                units=layer_width[i],
                activation=act_fun[i],
                input_shape=(input_dim,),
            ))

    # Add in a reshape to get (degree, output_dim)
    bezier_model.add(tf.keras.layers.Reshape((degree + 1, output_dim)))

    return bezier_model


def create_sequential_regularized_model(
        input_dim: int, output_dim: int,
        depth: int,
        regularizer, reg_penalty,
        layer_width=None, act_fun=None,
):
    """Create a simple sequential neural network with regularization.

    Description:
        Creates a simple tensorflow sequential neural network with specifiable widths. All activation
        functions are simply ReLU. Regularization is applied to the kernel and bias parameters.

    Args:
        input_dim (int): Input dimension i.e X.shape[1] where X is a numpy array and len(X.shape) == 2.
        output_dim (int): Output dimension i.e. Y.shape[2] where Y is a numpy array and
            len(Y.shape) == 3
        depth (int): Number of layers in the neural network.
        regularizer (callable): Tensorflow regularizer to call to be applied to the kernel and bias.
        reg_penalty (float): penalty associated with regularization in training.
        layer_width (tuple or int): Specifying individual layer widths. If None, all layers are taken
            to be the output dimension. If an integer, then all layers have that width. If a tuple,
            then each element is a layer width with the assertion that len(layer_width) == depth.
        act_fun (callable or tuple(callables)): Activation function to use on layers. Sample logic as
            layer_width but this time with an activation function. Default is the tensorflow relu
            function.

    Returns:
        regularized_model (tf.keras.Model): Sequential module with regularization applied to both the
            kernel and bias in each layer.
    """

    # Immediate assertions
    assert input_dim >= 0
    assert output_dim >= 0
    assert depth > 0

    # Do the layer width logic
    if layer_width is None:
        layer_width = (output_di,) * depth  # Simply match output width
    elif type(layer_width) == int:
        assert layer_width > 0
        layer_width = (layer_width,) * depth
    elif type(layer_width) == tuple:
        assert len(layer_width) == depth
        for lw in layer_width:
            assert type(lw) == int
            assert lw > 0
        assert layer_width[-1] == ouput_dim
    else:
        raise IOError(
            "layer_width needs to be None, tuple, or int rather than %s" % type(layer_width)
        )

    # Do the activation function logic
    if act_fun is None:
        act_fun = (tf.nn.relu,) * depth  # default 
    elif type(act_fun) == tuple:
        assert len(act_fun) == depth
        for af in act_fun:
            try:
                tf.keras.activations.serialize(af)
            except ValueError as ve:
                raise ValueError("One of the activation functions is invalid in tuple")
    else:
        tf.keras.activations.serialize(act_fun)
        act_fun = (act_fun,) * depth  # Match number of layers

    # TODO: do the same thing with regularization and penalty terms
    # Regularization function
    if type(regularizer) == tuple:
        assert len(regularizer) == depth
    else:
        regularizer = (regularizer,)*depth

    if type(reg_penalty) == tuple:
        assert len(reg_penalty) == depth
    else:
        reg_penalty = (reg_penalty,)*depth

    # Create model
    regularized_model = tf.keras.Sequential()
    
    for i in range(depth):
        # For all every other layer
        if i > 0:
            regularized_model.add(tf.keras.layers.Dense(
                units=layer_width[i],
                activation=act_fun[i],
                kernel_regularizer=regularizer[i](reg_penalty[i]),
                bias_regularizer=regularizer[i](reg_penalty[i]),
            ))
        # Input layer
        else:
            regularized_model.add(tf.keras.layers.Dense(
                units=layer_width[i],
                activation=act_fun[i],
                kernel_regularizer=regularizer[i](reg_penalty[i]),
                bias_regularizer=regularizer[i](reg_penalty[i]),
                input_shape=(input_dim,),
            ))

    return regularized_model

# Running as script
if __name__ == "__main__":
    # get script input
    epochs = 1 if len(sys.argv) == 1 else int(sys.argv[1])

    # Get training data
    test_df = rbd.cleaned_blackbird_test(*DEFAULT_TEST)
    X, Y, _, _ = rbd.generate_opt_control_test_data(test_df, 1, 1, DEFAULT_DOWNSAMPLING_DICT)

    # Removing unnecessary variables
    del test_df

    # Create neural network
    print("Input: (%i, %i)" % X.shape)
    print("Output: (%i, %i)" % Y.shape)
    model = create_sequential_model(input_dim=240, output_dim=40, depth=10)
    model.compile(optimizer='adam', loss=tf.keras.losses.mse)
    
    # Fit the model
    training_h = model.fit(
        x=X, y=Y,
        batch_size=1000,
        epochs=epochs,
        verbose=0
    )

    training_h = np.array(training_h.history['loss'])

    print("Minimum MSE=%f at epoch=%i" % (np.min(training_h), np.argmin(training_h)))

"""
File: TRAIN_OPTIMAL_CONTROLLER.PY
Author: Patrick McNamee
Date: Feb 3, 2021

Brief: Python file for training neural networks as optimal controllers.
"""


# Import appropriate python libraries
import os
import sys
import tensorflow as tf


# Add to custom codes to path
#  Current file location: EECS-Masters/src/models/
for folder in ['functions', 'models']:
    sys.path.append(os.path.abspath('../../%s' % folder))

# Additional imports
import read_blackbird_dataset as rbd
import custom_loss_functions as clf


TEST_FLIGHTS_LIST = [
    ()
]


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
    predictions

    # Loss function stuff
    tf.nn.softmax(predictions).numpy()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss_fn(y_train[:1], predictions).numpy()

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
    probability_model(x_test[:5])
    
    
# Running as script
if __name__ == "__main__":
    
    print('Successfully ran as script')

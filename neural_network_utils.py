# Defines the neural network model architecture and helper layers

import tensorflow as tf

class FeatureProviderLayer(tf.keras.layers.Layer):
    """
    A non-trainable layer that acts as a lookup table for features.
    It can be pre-loaded with a large feature set, and then fed with indices
    during training to fetch the corresponding feature vectors. This can improve
    training efficiency by minimizing data transfer to the GPU.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_features(self, features):
        # Loads the entire feature set as a constant tensor
        self.features = tf.constant(features)

    def call(self, index):
        # Fetches features corresponding to the input indices
        return tf.gather(self.features, index)

    def get_config(self):
        return super().get_config()

class ComplexToRealImagLayer(tf.keras.layers.Layer):
    """
    A layer that converts a tensor of complex numbers into a real-valued tensor
    by stacking the real and imaginary parts along a new last dimension.
    This allows standard dense layers to process complex-valued inputs.
    """

    def __init__(self, dtype):
        super(ComplexToRealImagLayer, self).__init__(dtype = dtype)

    def call(self, features):
        # Stacks the real and imaginary parts. Shape (..., C) -> (..., C, 2)
        return tf.stack([tf.math.real(features), tf.math.imag(features)], axis = -1)

def construct_model(input_shape, name, dtype = tf.complex64):
    """
    Constructs and returns the Keras sequential model for localization.
    """

    return tf.keras.models.Sequential([
        tf.keras.Input(shape = input_shape, name = "input", dtype = dtype),			# Defines the model's input layer
        ComplexToRealImagLayer(dtype = dtype),										# Convert complex input features to real-valued pairs
        tf.keras.layers.Flatten(),													# Flatten the multi-dimensional features into a single long vector
        tf.keras.layers.BatchNormalization(),										# A stack of Dense layers with Batch Normalization to learn patterns	
        tf.keras.layers.Dense(1024, activation = "relu"),							# The number of neurons decreases, forcing the network to find compact representations
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(512, activation = "relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation = "relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation = "relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation = "relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(2, activation = "linear"),							# The final output layer; 2 neurons for predicting a 2D position (x, y); 'linear' activation is used for regression tasks to output any real value
    ], name = name)

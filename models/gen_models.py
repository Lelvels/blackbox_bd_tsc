import keras
from keras import backend as K
import tensorflow as tf
import numpy as np
from utils.constants import SEED
np.random.seed(SEED)

def create_cnn_generator(input_shape):
    """
    Create a CNN generator model based on the provided configuration.
    
    Args:
        input_shape (tuple): Shape of the input data.
        nb_classes (int): Number of classes for classification.
        training_config (dict): Training configuration parameters.
        generator_config (dict): Generator configuration parameters.
    
    Returns:
        keras.Model: Compiled CNN generator model.
    """
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=128*input_shape[1], kernel_size=15, padding='same', name='conv1')(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation(activation='relu')(conv1)

    conv2 = keras.layers.Conv1D(filters=512*input_shape[1], kernel_size=21, padding='same', name='conv2')(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)

    conv3 = keras.layers.Conv1D(filters=1024, kernel_size=8, padding='same', name='conv3')(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)

    fc1 = keras.layers.Dense(512, activation='relu')(conv3)
    fc2 = keras.layers.Dense(input_shape[1], activation='tanh')(fc1)

    output_layer = fc2
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    return model

def create_dynamic_ampl_cnn_generator(input_shape, max_amplitude=0.2): # NEW: Added max_amplitude parameter
    """
    Create a CNN generator model that outputs both a pattern and a learnable amplitude.
    
    Args:
        input_shape (tuple): Shape of the input data.
        max_amplitude (float): The maximum possible value for the learned amplitude.
    
    Returns:
        keras.Model: Compiled CNN generator model with two outputs.
    """
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=128*input_shape[1], kernel_size=15, padding='same', name='conv1')(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation(activation='relu')(conv1)

    conv2 = keras.layers.Conv1D(filters=512*input_shape[1], kernel_size=21, padding='same', name='conv2')(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)

    conv3 = keras.layers.Conv1D(filters=1024, kernel_size=8, padding='same', name='conv3')(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)

    fc1 = keras.layers.Dense(512, activation='relu')(conv3)

    # --- Head for Pattern Generation ---
    pattern_output = keras.layers.Dense(input_shape[1], activation='tanh', name='pattern_output')(fc1)

    # --- NEW: Head for Amplitude Generation ---
    # This head learns a data-dependent amplitude for each sample.
    # We use GlobalAveragePooling1D to get a single feature vector per sample from the conv features.
    amplitude_features = keras.layers.GlobalAveragePooling1D()(conv3) 
    amplitude_hidden = keras.layers.Dense(32, activation='relu', name='amplitude_hidden')(amplitude_features)
    # Sigmoid activation squashes the output to [0, 1]
    amplitude_scaler = keras.layers.Dense(1, activation='sigmoid', name='amplitude_scaler')(amplitude_hidden)
    # Scale the output by max_amplitude. The Reshape is to ensure the dimensions are broadcastable later.
    amplitude_output = keras.layers.Lambda(lambda x: x * max_amplitude, name='amplitude_output')(amplitude_scaler)
    amplitude_output = keras.layers.Reshape((1, 1), name='amplitude_reshaped')(amplitude_output)

    # The model now has two outputs
    model = keras.models.Model(inputs=input_layer, outputs=[pattern_output, amplitude_output])

    return model    

def get_generator_by_name(input_shape, max_amplitude, generator_name="dynamic_ampl_cnn"):
    """
    Get the generator model based on the specified configuration.
    
    Args:
        input_shape (tuple): Shape of the input data.
        nb_classes (int): Number of classes for classification.
        training_config (dict): Training configuration parameters.
        generator_config (dict): Generator configuration parameters.
    
    Returns:
        keras.Model: Compiled generator model.
    """
    if generator_name == 'cnn':
        return create_cnn_generator(input_shape)
    elif generator_name == 'dynamic_ampl_cnn':
        return create_dynamic_ampl_cnn_generator(input_shape, max_amplitude)
    else:
        raise ValueError(f"Unsupported generator model name: {generator_name}")
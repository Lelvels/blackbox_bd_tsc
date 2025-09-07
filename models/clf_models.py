import os
import typing

import numpy as np
import pandas as pd

import tensorflow as tf
import keras
from keras import backend as K
from keras import layers, Model, ops

from utils.constants import SEED

# Set Keras backend
os.environ["KERAS_BACKEND"] = "tensorflow"

np.random.seed(SEED)

def create_fcn_model(input_shape, nb_classes):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation(activation='relu')(conv1)

    conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)

    conv3 = keras.layers.Conv1D(128, kernel_size=3, padding='same')(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)

    gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)
    logit_layer = keras.layers.Dense(nb_classes, name='logits')(gap_layer)
    output_layer = keras.layers.Activation('softmax', name='softmax_output')(logit_layer)
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    return model

def create_resnet_model(input_shape, nb_classes):
    # BLOCK 1
    n_feature_maps = 64
    input_layer = keras.layers.Input(input_shape)

    conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

    output_block_1 = keras.layers.add([shortcut_y, conv_z])
    output_block_1 = keras.layers.Activation('relu')(output_block_1)

    # BLOCK 2

    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

    output_block_2 = keras.layers.add([shortcut_y, conv_z])
    output_block_2 = keras.layers.Activation('relu')(output_block_2)

    # BLOCK 3

    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # no need to expand channels because they are equal
    shortcut_y = keras.layers.BatchNormalization()(output_block_2)
    output_block_3 = keras.layers.add([shortcut_y, conv_z])
    output_block_3 = keras.layers.Activation('relu')(output_block_3)

    # FINAL
    gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)
    logit_layer = keras.layers.Dense(nb_classes, name='logits')(gap_layer)
    output_layer = keras.layers.Activation('softmax', name='softmax_output')(logit_layer)
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    return model

def create_lstm_model(input_shape, nb_classes):
    input_layer = keras.layers.Input(input_shape)

    lstm_layer = keras.layers.LSTM(256, return_sequences=True)(input_layer)
    # lstm_layer = keras.layers.BatchNormalization()(lstm_layer)
    lstm_layer = keras.layers.Activation('relu')(lstm_layer)

    lstm_layer = keras.layers.LSTM(128)(lstm_layer)
    # lstm_layer = keras.layers.BatchNormalization()(lstm_layer)
    lstm_layer = keras.layers.Activation('relu')(lstm_layer)

    logit_layer = keras.layers.Dense(nb_classes, name='logits')(lstm_layer)
    output_layer = keras.layers.Activation('softmax', name='softmax_output')(logit_layer)
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    return model

def create_gru_model(input_shape, nb_classes):
    input_layer = keras.layers.Input(input_shape)

    gru_layer = keras.layers.GRU(256, return_sequences=True)(input_layer)
    gru_layer = keras.layers.BatchNormalization()(gru_layer)
    gru_layer = keras.layers.Activation('relu')(gru_layer)

    gru_layer = keras.layers.GRU(128)(gru_layer)
    gru_layer = keras.layers.BatchNormalization()(gru_layer)
    gru_layer = keras.layers.Activation('relu')(gru_layer)

    dense_layer = keras.layers.Dense(64, activation='relu')(gru_layer)
    logit_layer = keras.layers.Dense(nb_classes, name='logits')(dense_layer)
    output_layer = keras.layers.Activation('softmax', name='softmax_output')(logit_layer)
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    return model

def create_transformer_model(input_shape,
                             nb_classes,
                             head_size=128,
                             num_heads=4,
                             ff_dim=4,
                             num_transformer_blocks=4,
                             mlp_units=[64],
                             dropout=0.1,
                             mlp_dropout=0.1):
    def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout):
        # Attention and Normalization
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(inputs, inputs)
        x = layers.Dropout(dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs

        # Feed Forward Part
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        return x + res

    # Input layer
    input_layer = keras.layers.Input(input_shape)
    x = input_layer

    # Transformer blocks
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    
    # Global Average Pooling
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    
    # Output layer
    output_layer = layers.Dense(nb_classes, activation="softmax")(x)
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    return model

def get_model_by_name(model_name, input_shape, nb_classes) -> keras.Model:
    """
    Get the model by its name.
    
    Args:
        model_name (str): Name of the model to create.
        input_shape (tuple): Shape of the input data.
        nb_classes (int): Number of classes for classification.
    
    Returns:
        keras.Model: Compiled model.
    """
    if model_name == 'fcn':
        return create_fcn_model(input_shape, nb_classes)
    elif model_name == 'resnet':
        return create_resnet_model(input_shape, nb_classes)
    elif model_name == 'lstm':
        return create_lstm_model(input_shape, nb_classes)
    elif model_name == 'transformer':
        return create_transformer_model(input_shape, nb_classes)
    elif model_name == 'gru':
        return create_gru_model(input_shape, nb_classes)
    else:
        raise ValueError(f"Model {model_name} is not supported.")
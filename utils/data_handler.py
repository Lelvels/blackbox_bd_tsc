from builtins import print
import numpy as np
import pandas as pd
import matplotlib
import tensorflow as tf
import keras
from utils.constants import SEED
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def select_stratified_by_sample(x_data, y_data, no_samples, random_state=SEED):
    """
    Select a stratified random subset from the data based on max samples.
    
    Args:
        x_data: Input features
        y_data: Labels (one-hot encoded)
        max_samples: Maximum number of samples to select
        random_state: Random seed for reproducibility
    
    Returns:
        x_subset, y_subset: Selected subset of data
    """
    if no_samples >= len(x_data):
        return x_data, y_data
    
    # Calculate subset ratio
    subset_ratio = no_samples / len(x_data)
    
    # Convert one-hot encoded labels to class indices for stratification
    y_indices = np.argmax(y_data, axis=1)
    
    # Use train_test_split to get stratified subset
    x_subset, _, y_subset, _ = train_test_split(
        x_data, y_data,
        train_size=subset_ratio,
        stratify=y_indices,
        random_state=random_state
    )
    
    return x_subset, y_subset

def select_stratified_by_ratio(x_data, y_data, data_ratio, random_state=SEED):
    """
    Select a stratified random subset from the data based on a ratio.
    
    Args:
        x_data: Input features
        y_data: Labels (one-hot encoded)
        data_ratio: Fraction of data to retain (0 < data_ratio <= 1)
        random_state: Random seed for reproducibility
    
    Returns:
        x_subset, y_subset: Selected subset of data
    """
    if not (0 < data_ratio <= 1):
        raise ValueError("data_ratio must be in the range (0, 1]")
    
    # Calculate number of samples to select
    no_samples = int(len(x_data) * data_ratio)
    
    return select_stratified_by_sample(x_data, y_data, no_samples, random_state)

def get_tsc_train_dataset(dataset_name, data_type, data_ratio=1.0):
    """
    Get the training dataset for time series classification.

    Args:
        archive_name: Name of the archive to read from
        dataset_name: Name of the dataset to read
        univariate: Whether to use univariate data (default: False)
        data_ratio: Fraction of data to retain (default: 1.0)
        data_type: Type of model to train (surro or target)
    
    Returns:
        Tuple containing x_train, y_train, x_test, y_test
    """
    data_root_dir = "<path_to_dataset>"
    if dataset_name not in ["iAWE", "MotionSense", "VNDALE"]:
        raise ValueError("dataset_name must be either 'iAWE', 'MotionSense' or 'VNDALE'")

    if data_type not in ["sp", "atk"]:
        raise ValueError("data_type must be either 'sp' or 'atk'")

    x_train, y_train, x_test, y_test = None, None, None, None
    
    root_dir_dataset = f"{data_root_dir}/{dataset_name}/train_test_np/"
    x_train = np.load(root_dir_dataset + f'X_train_{data_type}.npy')
    y_train = np.load(root_dir_dataset + f'y_train_{data_type}.npy')
    x_test = np.load(root_dir_dataset + f'X_test_{data_type}.npy')
    y_test = np.load(root_dir_dataset + f'y_test_{data_type}.npy')

    if dataset_name == 'iAWE':
        # iAWE dataset has 5 channels, we only use 4 channels (0, 1, 3, 4)
        x_train = x_train[:, :, [0, 1, 3, 4]]
        x_test = x_test[:, :, [0, 1, 3, 4]]

    # Apply stratified data reduction if ratio < 1.0
    if data_ratio < 1.0:
        # Stratify training set
        unique_classes = np.unique(y_train)
        train_indices = []
        
        for cls in unique_classes:
            cls_indices = np.where(y_train == cls)[0]
            cls_samples = int(len(cls_indices) * data_ratio)
            if cls_samples > 0:
                selected_indices = np.random.choice(cls_indices, cls_samples, replace=False)
                train_indices.extend(selected_indices)
        
        train_indices = np.array(train_indices)
        x_train = x_train[train_indices]
        y_train = y_train[train_indices]
        
        # Stratify test set
        test_indices = []
        unique_test_classes = np.unique(y_test)
        
        for cls in unique_test_classes:
            cls_indices = np.where(y_test == cls)[0]
            cls_samples = int(len(cls_indices) * data_ratio)
            if cls_samples > 0:
                selected_indices = np.random.choice(cls_indices, cls_samples, replace=False)
                test_indices.extend(selected_indices)
        
        test_indices = np.array(test_indices)
        x_test = x_test[test_indices]
        y_test = y_test[test_indices]
        
        # Print statistics
        print(f"Dataset {dataset_name} reduced to {data_ratio*100:.1f}% with stratified sampling")
        print(f"Shape of x_train: {x_train.shape}, y_train: {y_train.shape}")
        print(f"Shape of x_test: {x_test.shape}, y_test: {y_test.shape}")

    # Cast types
    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_test = y_test.astype(np.float32)

    return x_train, y_train, x_test, y_test

def preprocess_data(x_train, y_train, x_test, y_test):
    """
    One-hot encode the labels and reshape the data if univariate.
    
    Args:
        x_train: Training data
        x_test: Test data
        univariate: Whether the data is univariate (default: False)
    
    Returns:
        Tuple containing preprocessed x_train and x_test
    """
    # Transform the labels from integers to one hot vectors
    enc = OneHotEncoder(categories='auto')
    print("y_train shape before encoding:", y_train.shape)
    print("y_test shape before encoding:", y_test.shape)
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    
    # Cast types to float32 for TensorFlow compatibility
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    return x_train, y_train, x_test, y_test, enc

def create_dataset(x_data, y_data, batch_size=64, shuffle=True, prefetch=True):
    """Create tf.data.Dataset from numpy arrays"""
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(x_data), 10000))
    
    dataset = dataset.batch(batch_size)
    
    if prefetch:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def create_prediction_dataset(x_data, batch_size=512, prefetch=True):
    """Create dataset for prediction (no labels)"""
    dataset = tf.data.Dataset.from_tensor_slices(x_data)
    dataset = dataset.batch(batch_size)
    
    if prefetch:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def predict_with_dataset(model: keras.Model, dataset):
    """Efficient prediction using tf.data.Dataset"""
    predictions = []
    
    for batch in dataset:
        batch_pred = model.predict_on_batch(batch)
        predictions.append(batch_pred)
    
    return np.concatenate(predictions, axis=0)

def evaluate_with_dataset(model: keras.Model, dataset):
    """Efficient evaluation using tf.data.Dataset"""
    return model.evaluate(dataset, verbose=0)
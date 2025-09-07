import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Allow GPU
os.environ['KERAS_BACKEND'] = 'tensorflow'  # Use TensorFlow backend

# Other imports
import numpy as np
import pandas as pd
from builtins import print
from utils.constants import SEED
import gc
import tensorflow as tf
from keras import backend as K
import keras
import logging
from models.clf_wrapper import ClassifierWrapper
from models.trigger_gen import TriggerGenerator
from numba import cuda
import csv

# Set random seeds for reproducibility
np.random.seed(SEED)

class SimpleBudgetCounter:
    """Simple budget counter for blackbox attacks"""
    def __init__(self):
        self.query_count = 0
    
    def query(self, number: int) -> bool:
        """
        Record a query and check if budget is exceeded
        
        Returns:
            True if budget allows more queries, False otherwise
        """
        self.query_count += number
        return True
    
    def get_count(self) -> int:
        """Get current query count"""
        return self.query_count
    
    def reset(self):
        """Reset the counter"""
        self.query_count = 0

def zscore_transform(data, save_path):
    """
    Apply Z-score normalization to 3D data and save parameters
    
    Args:
        data: 3D numpy array with shape (n_samples, seq_len, n_features)
        save_path: Path to save mean/std parameters
    
    Returns:
        normalized_data: Z-score normalized data with same shape as input
    """
    print(f"Input data shape: {data.shape}")
    
    # Calculate mean and std across samples and time steps for each feature
    # Shape: (n_features,)
    mean_vals = data.mean(axis=(0, 1))
    std_vals = data.std(axis=(0, 1))
    
    print(f"Mean values per feature: {mean_vals}")
    print(f"Std values per feature: {std_vals}")
    
    # Avoid division by zero (if std is 0, keep original values)
    std_vals[std_vals == 0] = 1.0
    
    # Z-score normalization: (x - mean) / std
    normalized_data = (data - mean_vals) / std_vals
    
    # Save parameters
    np.savez(save_path, mean_vals=mean_vals, std_vals=std_vals)
    print(f"Z-score parameters saved to: {save_path}")
    
    print(f"Normalized data mean: {normalized_data.mean(axis=(0,1))}")
    print(f"Normalized data std: {normalized_data.std(axis=(0,1))}")
    print(f"Normalized data range: [{normalized_data.min():.6f}, {normalized_data.max():.6f}]")
    
    return normalized_data

def zscore_inverse_transform(normalized_data, params_path):
    """
    Inverse Z-score normalization using saved parameters
    
    Args:
        normalized_data: Normalized 3D numpy array with shape (n_samples, seq_len, n_features)
        params_path: Path to load mean/std parameters
    
    Returns:
        original_data: Data in original scale
    """
    print(f"Normalized data shape: {normalized_data.shape}")
    
    # Load parameters
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Parameters file not found: {params_path}")
    
    params = np.load(params_path)
    mean_vals = params['mean_vals']
    std_vals = params['std_vals']
    
    print(f"Loaded mean values: {mean_vals}")
    print(f"Loaded std values: {std_vals}")
    
    # Inverse transform: normalized_data * std + mean
    original_data = normalized_data * std_vals + mean_vals
    
    print(f"Inverse transformed data range: [{original_data.min():.6f}, {original_data.max():.6f}]")
    
    return original_data

def check_normalization_quality(original_data, normalized_data, params_path):
    """
    Check the quality of normalization
    """
    # Load parameters
    params = np.load(params_path)
    mean_vals = params['mean_vals']
    std_vals = params['std_vals']
    
    # Check if normalized data has mean~0 and std~1
    actual_mean = normalized_data.mean(axis=(0, 1))
    actual_std = normalized_data.std(axis=(0, 1))
    
    print("=== Normalization Quality Check ===")
    print(f"Expected mean: {np.zeros_like(mean_vals)}")
    print(f"Actual mean: {actual_mean}")
    print(f"Mean error: {np.abs(actual_mean).max():.10f}")
    
    print(f"\nExpected std: {np.ones_like(std_vals)}")
    print(f"Actual std: {actual_std}")
    print(f"Std error: {np.abs(actual_std - 1).max():.10f}")
    
    # Test inverse transform
    reconstructed = zscore_inverse_transform(normalized_data, params_path)
    reconstruction_error = np.mean(np.abs(original_data - reconstructed))
    print(f"\nReconstruction error: {reconstruction_error:.10f}")
    
    return actual_mean, actual_std, reconstruction_error

def clear_memory_safe():
    """Clear memory safely without affecting loaded models"""
    # Only clear Python garbage collection
    gc.collect()
    
    # Clear any cached operations but keep models intact
    keras.utils.clear_session()

def limit_gpu_memory():
    """Limit GPU memory growth"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

def reset_tensorflow():
    """Reset TensorFlow session"""
    try:
        # Clear Keras session
        keras.backend.clear_session()
        # Reset TensorFlow graph
        tf.compat.v1.reset_default_graph()
        gc.collect()
    except Exception as e:
        print(f"Error resetting TensorFlow: {e}")


def get_last_completed_epoch(csv_path, main_epochs, atk_epochs):
    """
    Determine the last completed epoch from CSV file
    Returns: (last_main_epoch, last_atk_epoch, should_resume)
    """
    if not os.path.exists(csv_path):
        return 0, 0, False
    
    try:
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            return 0, 0, False
        
        # Get the last row
        last_row = df.iloc[-1]
        last_main_epoch = int(last_row['Main Epoch'])
        last_atk_epoch = int(last_row['Attack Epoch'])
        
        # Check if we need to resume from next attack epoch or next main epoch
        if last_atk_epoch < atk_epochs:
            # Resume from next attack epoch in same main epoch
            return last_main_epoch, last_atk_epoch, True
        elif last_main_epoch < main_epochs:
            # Resume from next main epoch
            return last_main_epoch, atk_epochs, True
        else:
            # Training is complete
            return last_main_epoch, last_atk_epoch, False
            
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return 0, 0, False
    
def load_model_if_exists(model_wrapper: ClassifierWrapper, weights_path, logger):
    """Load model weights if they exist, otherwise use default weights"""
    if os.path.exists(weights_path):
        model_wrapper.model.load_weights(weights_path)
        logger.info(f"Loaded model weights from: {weights_path}")
        return True
    else:
        logger.warning(f"Model weights not found at: {weights_path}")
        return False

def clear_memory_and_reload_model(model_wrapper: ClassifierWrapper, weights_path, logger: logging.Logger):
    """
    Clear memory and reload model from weights file
    """
    try:
        # Save current weights if the path doesn't exist
        if not os.path.exists(weights_path):
            model_wrapper.model.save_weights(weights_path)
            logger.info(f"[Memory Management] Saved model weights to {weights_path}")

        # Log the extraction of model parameters
        logger.info("[Memory Management] Extracting model parameters for reloading")

        # Extract model parameters
        clf_name = model_wrapper.clf_name
        input_shape = model_wrapper.input_shape
        nb_classes = model_wrapper.nb_classes
        training_config = model_wrapper.training_config
        custom_loss = model_wrapper.custom_loss
        output_directory = model_wrapper.output_directory
        verbose = model_wrapper.verbose

        # delete the current model to free memory
        del model_wrapper
        
        # Clear session and memory
        reset_tensorflow()

        # Rebuild the model
        new_model_wrapper = ClassifierWrapper(
            output_directory=output_directory,
            input_shape=input_shape,
            nb_classes=nb_classes,
            training_config=training_config,
            clf_name=clf_name,
            verbose=verbose,
            custom_loss=custom_loss
        )
        
        # Load weights back
        new_model_wrapper.model.load_weights(weights_path)
        logger.info(f"[Memory Management] Reloaded model from {weights_path}")

        return new_model_wrapper
    except Exception as e:
        logger.error(f"[Memory Management Error] Failed to clear and reload model: {e}")
        raise e
    
def clear_memory_and_reload_generator(trigger_gen_wrapper: TriggerGenerator, weights_path, logger: logging.Logger):
    """
    Clear memory and reload generator from weights file
    """
    try:
        # Save current weights if the path doesn't exist
        if not os.path.exists(weights_path):
            trigger_gen_wrapper.generator.save_weights(weights_path)
            logger.info(f"[Memory Management] Saved generator weights to {weights_path}")

        # Log the extraction of model parameters
        logger.info("[Memory Management] Extracting generator parameters for reloading")
        output_directory = trigger_gen_wrapper.output_directory
        generator_config = trigger_gen_wrapper.generator_config
        max_amplitude = trigger_gen_wrapper.max_amplitude
        input_shape = trigger_gen_wrapper.input_shape
        enc = trigger_gen_wrapper.enc
        gen_type = trigger_gen_wrapper.gen_type
        
        # Clear session and memory
        reset_tensorflow()
        # delete the current generator to free memory
        del trigger_gen_wrapper
        
        # Rebuild the generator
        new_trigger_gen_wrapper = TriggerGenerator(
            output_directory=output_directory,
            generator_config=generator_config,
            max_amplitude=max_amplitude,
            input_shape=input_shape,
            gen_type=gen_type,
            enc=enc
        )
        
        # Load weights back
        new_trigger_gen_wrapper.generator.load_weights(weights_path)
        logger.info(f"[Memory Management] Reloaded generator from {weights_path}")

        return new_trigger_gen_wrapper
    except Exception as e:
        logger.error(f"[Memory Management Error] Failed to clear and reload generator: {e}")
        raise e

def safe_model_predict(model, data, batch_size=512, logger: logging.Logger = None):
    """
    Safe prediction with memory management
    """
    try:
        if hasattr(data, '__iter__') and not isinstance(data, (str, bytes)):
            # If data is a dataset
            predictions = model.predict(data, verbose=0)
        else:
            # If data is a numpy array
            predictions = model.predict(data, batch_size=batch_size, verbose=0)
        return predictions
    except Exception as e:
        if logger:
            logger.warning(f"[Memory Warning] Prediction failed, clearing memory and retrying: {e}")
        keras.backend.clear_session()
        gc.collect()
        if hasattr(data, '__iter__') and not isinstance(data, (str, bytes)):
            predictions = model.predict(data, verbose=0)
        else:
            predictions = model.predict(data, batch_size=batch_size, verbose=0)
        return predictions

def initialize_csv_logger(csv_path, headers):
    """Initialize CSV file with headers"""
    if os.path.exists(csv_path):
        # If file exists, keep using it
        return
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

def log_to_csv(csv_path, data):
    """Append data row to CSV file"""
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)

def setup_logger(file_name, log_file=None):
    logger = logging.getLogger(file_name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger